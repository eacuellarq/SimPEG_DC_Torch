"""
Torch optimizers exposed as stock-SimPEG (>=0.25) Optimizations.

Two of them, sharing one objective:

- ``TorchLBFGS``: one Minimize *iteration* = one inner ``torch.optim.LBFGS``
  solve at FIXED beta.
- ``TorchGaussNewton``: one Minimize iteration = one Gauss-Newton step, its
  normal equations solved by truncated, Jacobi-preconditioned CG on the exact
  J v / J^T v of the engine, followed by a quadratic-interpolation line search.

Between iterations the standard directives run unchanged (``BetaSchedule`` /
``TargetMisfit`` / ``UpdateIRLS`` / ``SaveOutputEveryIteration``), seeing
``invProb.phi_d/phi_m/beta`` exactly as with any other optimizer.

The objective is torch throughout:
- phi_d from a dctorch engine (``engine.dpred`` is autograd-traceable), using
  the 0.25 convention ``||W r||^2`` (NO 1/2 factor) — Student-t if the data
  misfit exposes ``nu``;
- phi_m by affine extraction of the SimPEG regularization (each leaf is
  ``mult * ||W_j (A_j m + b_j)||^2`` with A_j, b_j constant), weights re-read
  every iteration so IRLS updates enter for free, asserted against ``reg(m)``.

Ported from the 0.18-fork ``optimization.TorchLBFGS`` (validated corr=1.0000
against the production scripts); differences here: explicit ``engine=`` (no
global config), 0.25 factor conventions, ``bfgsH0`` stubbed so
``BaseInvProblem.startup`` skips factorizing ``reg.deriv2``.
"""
import numpy as np
import scipy.sparse as sp
from simpeg.optimization import Minimize, Remember
from simpeg.utils import call_hooks, timeIt


class _IdentityH0:
    """Placeholder so BaseInvProblem.startup skips the bfgsH0 factorization."""

    def __mul__(self, x):
        return x

    matvec = solve = __mul__


class _TorchMinimize(Minimize, Remember):
    """What every torch optimizer here shares: the SimPEG protocol, phi_d and
    phi_m in torch, and the memoized objective at the accepted iterate.

    Subclasses provide ``findSearchDirection``; it must leave in ``self._post``
    the objective at the model it accepts (``_eval_full`` does that), so that
    ``modifySearchDirection`` takes the step without evaluating it again.
    """

    engine = None          #: dctorch engine with a differentiable .dpred(m_t)
    model_device = None    #: device for the model tensor (None = engine.device)
    log_data = False       #: misfit on ln|d| (log apparent resistivity, the
                           #: geometric factor cancels): r = (ln|d| - ln|dobs|)
                           #: / relative error, the relative error being
                           #: std / |dobs| from the data misfit's W
    lower = None           #: optional bounds: differentiable clamp in the
    upper = None           #: closure + np.clip in projection

    # stopping is governed by directives (+ maxIter): keep Minimize stoppers loose
    tolF = 1e-8
    tolX = 1e-10
    tolG = 1e-10
    eps = 1e-16

    def __init__(self, **kwargs):
        Minimize.__init__(self, **kwargs)
        self.bfgsH0 = _IdentityH0()

    def _startup_torch_common(self, x0):
        self._torch_ready = False
        self._phi_m_terms = None
        self._post = None

    # ------------------------------------------------------------------ #
    #  torch objective pieces
    # ------------------------------------------------------------------ #
    def _torch_setup(self):
        if self._torch_ready:
            return
        import torch

        if self.engine is None:
            raise ValueError(f"{type(self).__name__} requires engine=<dctorch engine>")
        invProb = self.parent
        objfcts = invProb.dmisfit.objfcts
        if len(objfcts) != 1:
            raise NotImplementedError(
                f"{type(self).__name__} currently supports a single data-misfit term")
        dm = objfcts[0]
        self._mdev = self.model_device or self.engine.device
        # protocol-side tensors (model, phi_m, W, dobs) stay float64 regardless
        # of the engine precision: the engine casts internally and returns d in
        # the model dtype, and the phi_m==reg(m) assert needs f64 exactness
        self._dt = torch.float64
        self._W_t = torch.as_tensor(np.asarray(dm.W.diagonal()),
                                    dtype=self._dt, device=self._mdev)
        self._dobs_t = torch.as_tensor(np.asarray(dm.data.dobs, dtype=float),
                                       dtype=self._dt, device=self._mdev)
        self._nu = float(getattr(dm, "nu", 0.0) or 0.0)
        if self.log_data:
            if bool((self._dobs_t == 0).any()):
                raise ValueError("log_data needs non-zero observed data")
            self._sgn_t = torch.sign(self._dobs_t)
            self._Wl_t = self._W_t * self._dobs_t.abs()      # 1 / relative error
            self._ldobs_t = torch.log(self._dobs_t.abs())
        self._torch_ready = True

    def _residual(self, d):
        """Weighted residual: W (d - dobs), or (ln|d| - ln|dobs|) / rel. error."""
        import torch

        if self.log_data:
            sd = torch.clamp(self._sgn_t * d, min=1e-300)  # a sign flip -> huge misfit
            return self._Wl_t * (torch.log(sd) - self._ldobs_t)
        return self._W_t * (d - self._dobs_t)

    def _lin_weight(self, d):
        """dr/dd at d: W, or W_log / d for the log misfit (d ln|d| / dd = 1/d)."""
        if self.log_data:
            return self._Wl_t / d
        return self._W_t

    def _phi_d_of_r(self, r):
        """phi_d of the weighted residual: ||r||^2, or Student-t(nu)."""
        import torch

        if self._nu > 0:
            return (self._nu + 1.0) * torch.sum(torch.log1p(r**2 / self._nu))
        return torch.sum(r**2)

    def _phi_d_t(self, m_t):
        """(phi_d, dpred): ||W r||^2 (0.25 convention) or Student-t(nu)."""
        d = self.engine.dpred(m_t)
        if d.device != self._W_t.device:
            d = d.to(self._W_t.device)
        return self._phi_d_of_r(self._residual(d)), d

    def _build_phi_m_t(self):
        """Affine extraction of the (possibly IRLS-weighted) regularization:
        each leaf is mult * ||W_j (A_j m + b_j)||^2 (0.25: no 1/2), asserted
        against reg(m) at the current model."""
        import torch

        invProb = self.parent
        dev, dt = self._mdev, self._dt
        m_np = np.asarray(self.xc, dtype=float)

        leaves = []

        def walk(fct, mult):
            sub = getattr(fct, "objfcts", None)
            if sub:
                mults = getattr(fct, "multipliers", [1.0] * len(sub))
                for mi, fi in zip(mults, sub):
                    walk(fi, mult * mi)
            else:
                leaves.append((mult, fct))

        walk(invProb.reg, 1.0)

        if self._phi_m_terms is None or len(self._phi_m_terms) != len(leaves):
            terms = []
            self._phi_m_A_np = {}          # scipy copies, keyed by id(A_t)
            probe = m_np + 0.3 * np.random.RandomState(5).randn(m_np.size)
            for mult, fct in leaves:
                A = sp.csr_matrix(fct.f_m_deriv(probe))
                b = np.asarray(fct.f_m(probe)) - A @ probe
                Ac = A.tocoo()
                A_t = torch.sparse_coo_tensor(
                    torch.tensor(np.vstack((Ac.row, Ac.col)),
                                 dtype=torch.int64, device=dev),
                    torch.tensor(Ac.data, dtype=dt, device=dev),
                    Ac.shape).coalesce()
                b_t = torch.as_tensor(b, dtype=dt, device=dev)
                terms.append([fct, A_t, b_t, None])
                self._phi_m_A_np[id(A_t)] = A
            self._phi_m_terms = terms

        mults_t = []
        for (mult, fct), term in zip(leaves, self._phi_m_terms):
            term[3] = torch.as_tensor(np.asarray(fct.W.diagonal()),
                                      dtype=dt, device=dev)
            mults_t.append(mult)
        self._phi_m_mults = mults_t

        terms = self._phi_m_terms

        def phi_m_t(m_t):
            val = m_t.new_zeros(())
            for mult, (fct, A_t, b_t, w_t) in zip(mults_t, terms):
                r = w_t * (torch.sparse.mm(A_t, m_t.unsqueeze(1)).squeeze(1)
                           + b_t)
                val = val + mult * torch.sum(r**2)
            return val

        with torch.no_grad():
            ours = float(phi_m_t(torch.as_tensor(m_np, dtype=dt, device=dev)))
        simpegs = float(invProb.reg(m_np))
        if abs(ours - simpegs) > 1e-8 * max(1.0, abs(simpegs)):
            raise AssertionError(
                f"phi_m mismatch: torch {ours!r} vs simpeg {simpegs!r} — the "
                "regularization is not an affine-quadratic the extraction "
                "supports")
        return phi_m_t

    # ------------------------------------------------------------------ #
    #  Minimize protocol
    # ------------------------------------------------------------------ #
    def minimize(self, evalFunction, x0):
        """Memoized objective at the accepted iterate; misses evaluate via
        _eval_full (pure torch) — NEVER invProb.evalFunction (numpy Jtvec)."""

        def wrapped(x, return_g=True, return_H=True):
            post = self._post
            if post is None or x is not post["m"]:
                post = self._eval_full(x)
            out = (post["f"],)
            if return_g:
                out += (post["g"],)
            if return_H:
                out += (None,)
            return out if len(out) > 1 else out[0]

        return Minimize.minimize(self, wrapped, x0)

    def _eval_full(self, x):
        """Objective, gradient and data at x, at the current beta. The data
        and regularization gradients are kept apart as well: beta is cooled
        between iterations, and a step taken at the new beta needs
        g_d + beta_new * g_m, not the stored g."""
        import torch

        self._torch_setup()
        phi_m_t = self._build_phi_m_t()
        invProb = self.parent
        beta = float(invProb.beta)
        x = np.asarray(x, dtype=float)

        m_t = torch.tensor(x, dtype=self._dt, device=self._mdev,
                           requires_grad=True)
        fd, d = self._phi_d_t(m_t)
        g_d, = torch.autograd.grad(fd, m_t)
        m2 = m_t.detach().requires_grad_(True)
        pm = phi_m_t(m2)
        g_m, = torch.autograd.grad(pm, m2)
        phi_d = float(fd)
        phi_m = float(invProb.reg(x))
        g_d = g_d.detach().cpu().numpy()
        g_m = g_m.detach().cpu().numpy()
        post = {
            "m": x,
            "f": phi_d + beta * phi_m,
            "phi_d": phi_d,
            "phi_m": phi_m,
            "dpred": d.detach().cpu().numpy(),
            "g": g_d + beta * g_m,
            "g_d": g_d,
            "g_m": g_m,
        }
        self._post = post
        invProb.model = x
        invProb.phi_d, invProb.phi_d_last = phi_d, invProb.phi_d
        invProb.phi_m, invProb.phi_m_last = phi_m, invProb.phi_m
        invProb.dpred = post["dpred"]
        return post

    @timeIt
    def modifySearchDirection(self, p):
        """No outer line search: the accepted iterate is the one
        findSearchDirection already evaluated."""
        self._LS_t = 1.0
        self.iterLS = 0
        xt = self.projection(self.xc + p)
        post = self._post
        if post is not None and np.allclose(xt, post["m"], rtol=1e-12,
                                            atol=1e-300):
            xt = post["m"]
            invProb = self.parent
            invProb.model = xt
            invProb.phi_d, invProb.phi_d_last = post["phi_d"], invProb.phi_d
            invProb.phi_m, invProb.phi_m_last = post["phi_m"], invProb.phi_m
            invProb.dpred = post["dpred"]
            self._LS_ft = post["f"]
        else:
            self._LS_ft = self._eval_full(xt)["f"]
            xt = self._post["m"]
        self._LS_xt = xt
        return self._LS_xt, True

    @call_hooks("projection")
    def projection(self, p):
        if self.lower is None and self.upper is None:
            return p
        return np.clip(
            p,
            -np.inf if self.lower is None else float(self.lower),
            np.inf if self.upper is None else float(self.upper))


class TorchLBFGS(_TorchMinimize):
    name = "Torch L-BFGS (dctorch)"

    maxIter = 40           #: outer iterations (beta levels)
    inner_maxiter = 30     #: max closures per inner torch LBFGS start
    inner_restarts = 2     #: fresh-history LBFGS starts per outer iteration
    history_size = 12      #: L-BFGS memory of the inner solver
    lr = 1.0               #: torch LBFGS learning rate (1.0 with strong_wolfe)
    line_search_fn = "strong_wolfe"
    persist_inner = True   #: ONE torch LBFGS across outers — measured 2x on
                           #: a field line (20s vs 39s, 255 vs 524 closures, best
                           #: corr): curvature survives the beta cooling.
                           #: Set False to restore fresh restarts per level.
    exact_post = True      #: no-grad forward at the accepted iterate (cheap 2.5D)

    def __init__(self, **kwargs):
        _TorchMinimize.__init__(self, **kwargs)
        self.printers.append(
            {"title": "#inner", "value": lambda M: M._inner_closures,
             "width": 8, "format": lambda v: f"{v:8d}"})

    def _startup_TorchLBFGS(self, x0):
        self._inner_closures = 0
        self._m_persist = None
        self._opt_persist = None

    @timeIt
    def findSearchDirection(self):
        import torch

        self._torch_setup()
        phi_m_t = self._build_phi_m_t()
        invProb = self.parent
        beta = float(invProb.beta)

        if self.persist_inner and self._m_persist is not None:
            m = self._m_persist
            with torch.no_grad():
                m.copy_(torch.as_tensor(np.asarray(self.xc, dtype=float),
                                        dtype=self._dt, device=self._mdev))
        else:
            m = torch.tensor(np.asarray(self.xc, dtype=float), dtype=self._dt,
                             device=self._mdev, requires_grad=True)
        if self.lower is not None or self.upper is not None:
            def bind(x):
                return torch.clamp(
                    x,
                    -np.inf if self.lower is None else float(self.lower),
                    np.inf if self.upper is None else float(self.upper))
        else:
            def bind(x):
                return x
        self._inner_closures = 0
        n_starts = 1 if self.persist_inner else self.inner_restarts
        for _ in range(n_starts):
            if self.persist_inner:
                if self._opt_persist is None:
                    self._opt_persist = torch.optim.LBFGS(
                        [m], lr=self.lr, max_iter=self.inner_maxiter,
                        history_size=self.history_size,
                        line_search_fn=self.line_search_fn)
                    self._m_persist = m
                opt = self._opt_persist
            else:
                opt = torch.optim.LBFGS(
                    [m], lr=self.lr, max_iter=self.inner_maxiter,
                    history_size=self.history_size,
                    line_search_fn=self.line_search_fn)

            def closure():
                opt.zero_grad()
                mc = bind(m)
                fd_c, d_c = self._phi_d_t(mc)
                loss = fd_c + beta * phi_m_t(mc)
                loss.backward()
                self._closure_last = (float(fd_c), d_c.detach())
                self._inner_closures += 1
                return loss

            opt.step(closure)

        m_np = bind(m).detach().cpu().numpy()
        if self.exact_post:
            with torch.no_grad():
                fd, d = self._phi_d_t(bind(m).detach())
            phi_d, d_post = float(fd), d
        else:
            phi_d, d_post = self._closure_last
        phi_m = float(invProb.reg(m_np))
        g_np = (m.grad.detach().cpu().numpy() if m.grad is not None
                else np.zeros_like(m_np))
        self._post = {
            "m": m_np,
            "f": phi_d + beta * phi_m,
            "phi_d": phi_d,
            "phi_m": phi_m,
            "dpred": d_post.detach().cpu().numpy(),
            "g": g_np,
        }
        return m_np - self.xc


def _pcg(apply_H, b, Minv, maxiter, rtol):
    """Preconditioned conjugate gradients on H x = b, from x = 0.

    Truncated on purpose: a Gauss-Newton step only needs the direction to the
    accuracy the linearization itself has, so the solve stops at a relative
    residual of `rtol` (RES2DINV's 'incomplete Gauss-Newton' uses 0.005-0.05).
    Stops early as well on non-positive curvature, which only round-off can
    produce for this H. Returns (x, iterations).
    """
    import torch

    x = torch.zeros_like(b)
    r = b.clone()
    z = Minv * r
    p = z.clone()
    rz = float(r @ z)
    bnorm = float(torch.linalg.norm(b))
    if bnorm == 0.0:
        return x, 0
    it = 0
    for it in range(1, maxiter + 1):
        Hp = apply_H(p)
        pHp = float(p @ Hp)
        if pHp <= 0.0:
            break
        alpha = rz / pHp
        x += alpha * p
        r -= alpha * Hp
        if float(torch.linalg.norm(r)) <= rtol * bnorm:
            break
        z = Minv * r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, it




class TorchGaussNewton(_TorchMinimize):
    """Gauss-Newton on the exact sensitivities of a dctorch engine.

    Each SimPEG iteration is one step: at the current beta,

        (J^T D J + beta H_m) p = -g,      g = J^T (W dphi/dr) + beta g_m

    with D = 2 W^2 for the L2 misfit (the 0.25 convention has no 1/2) and the
    IRLS weight 2 (nu+1) W^2 / (nu + r^2) for Student-t; H_m = sum_j 2 mult_j
    A_j^T W_j^2 A_j is the exact Hessian of the extracted regularization.

    ONE sensitivity evaluation per step serves both the gradient and the
    system: the line search leaves the accepted model with its data but no
    gradient, and the J (or linearization) the next step needs anyway
    supplies J^T (W dphi/dr). No separate autograd pass per iteration.

    Sensitivities, chosen by size (``jacobian``):

    - ``"explicit"``: J formed once per step by the batched adjoint
      (``engine.Jmatrix``). The regime of a 2-D line, where J is tens of MB.
    - ``"matrix_free"``: the forward linearized once per step
      (``engine.linearize``: one factorization, fields kept); every product is
      one batched solve, nothing of size nD x nM is stored.
    - ``"auto"`` (default): explicit when J fits ``explicit_max_gib``.

    Inner solve (``inner``):

    - ``"direct"``: the EXACT step, solved in data space. With B = beta H_m,
      Woodbury gives

          p = -(u - K S^-1 J u),   u = B^-1 g,  K = B^-1 J^T,
          S = D^-1 + J K   (nD x nD, dense Cholesky)

      i.e. one sparse factorization of H_m (model-independent structure, no
      PDE) and a dense system the size of the DATA, not the model -- where
      RES2DINV's direct path factorizes the nM x nM normal equations. Needs
      the explicit J and H_m positive definite (alpha_s > 0).
    - ``"cg"``: truncated CG (``cg_rtol``, RES2DINV's 'incomplete
      Gauss-Newton' uses 0.005-0.05) with a Jacobi preconditioner built from
      the exact diagonal of J^T D J. Works matrix-free.
    - ``"auto"`` (default): direct when possible and nD <= ``direct_max_data``,
      CG otherwise.

    The step is capped so no cell moves by more than ``max_step`` in
    log-resistivity (RES2DINV clamps its update the same way), then a line
    search uses what Gauss-Newton already knows: f(0), the slope g.p and f(1)
    give a parabola whose minimum is the next trial when the full step fails
    the Armijo test. One forward per trial, no gradients.

    Bounds (``lower``/``upper``) are handled by projection; cells on a bound
    with the gradient pushing outward are frozen for the step (CG only: the
    direct solve then falls back to CG).
    """

    name = "Torch Gauss-Newton (dctorch)"

    maxIter = 20           #: Gauss-Newton steps (one per beta level)
    jacobian = "auto"      #: "explicit" | "matrix_free" | "auto"
    explicit_max_gib = 1.0  #: J budget for the explicit path under "auto"
    jmatrix_block = 16     #: data per adjoint batch when J must come from
                           #: engine.Jmatrix (engines without `linearize`)
    inner = "cg"           #: "cg" | "direct" | "auto"
    direct_max_data = 8000  #: largest nD for the data-space direct solve
    cg_maxiter = 40        #: CG iterations per step
    cg_rtol = 1e-2         #: relative residual that ends a CG solve
    precond = "jacobi"     #: "jacobi" (exact diag of the GN Hessian) | "none"
    beta_search = True     #: Occam: choose beta inside every step (see _occam);
                           #: the default -- drop BetaSchedule from the
                           #: directives. False restores one step per beta
                           #: level with a schedule driving beta.
    chi_target = 1.0       #: chi^2 = phi_d / nD the beta search aims at
    beta_aim = 0.8         #: aim the LINEARIZED prediction at beta_aim * target:
                           #: the prediction is optimistic, < 1 lands sooner
    beta_span = (10.0, 0.1)  #: candidate betas, as factors of the current one
    beta_points = 9        #: candidates on that log-spaced grid
    max_step = 2.0         #: cap on |dm| per cell per step (log units); None = off
    ls_maxiter = 6         #: line-search trials (forwards) per step
    ls_c1 = 1e-4           #: Armijo constant

    def __init__(self, **kwargs):
        _TorchMinimize.__init__(self, **kwargs)
        self.printers.extend([
            {"title": "inner", "value": lambda M: M._inner_tag,
             "width": 7, "format": lambda v: f"{v:>7s}"},
            {"title": "step", "value": lambda M: M._ls_step,
             "width": 7, "format": lambda v: f"{v:7.3f}"},
        ])

    def _startup_TorchGaussNewton(self, x0):
        self._inner_tag = ""
        self._ls_step = 0.0
        self._ls_ok = True
        self._mode = None
        self._sens = None
        self.counts = {"cg": 0, "direct": 0, "occam": 0, "ls_forwards": 0,
                       "jacobians": 0, "linearizations": 0}

    # ------------------------------------------------------------------ #
    #  objective without gradients; the gradient comes with J
    # ------------------------------------------------------------------ #
    def minimize(self, evalFunction, x0):
        def wrapped(x, return_g=True, return_H=True):
            post = self._post
            if post is None or x is not post["m"]:
                post = self._eval_point(x)
            if return_g and post.get("g") is None:
                self._fill_gradient(post)
            out = (post["f"],)
            if return_g:
                out += (post["g"],)
            if return_H:
                out += (None,)
            return out if len(out) > 1 else out[0]

        return Minimize.minimize(self, wrapped, x0)

    def _eval_point(self, x, d=None):
        """phi_d, phi_m and f at x, at the current beta: one forward (none if
        the data d at x are given). The gradient is left for J to supply."""
        import torch

        self._torch_setup()
        self._build_phi_m_t()
        invProb = self.parent
        beta = float(invProb.beta)
        x = np.asarray(x, dtype=float)
        with torch.no_grad():
            if d is None:
                d = self.engine.dpred(torch.as_tensor(x, dtype=self._dt,
                                                      device=self._mdev))
            d = d.to(device=self._mdev, dtype=self._dt)
            phi_d = float(self._phi_d_of_r(self._residual(d)))
        phi_m = float(invProb.reg(x))
        post = {"m": x, "f": phi_d + beta * phi_m, "phi_d": phi_d,
                "phi_m": phi_m, "dpred": d.cpu().numpy(), "g": None}
        self._post = post
        invProb.model = x
        invProb.phi_d, invProb.phi_d_last = phi_d, invProb.phi_d
        invProb.phi_m, invProb.phi_m_last = phi_m, invProb.phi_m
        invProb.dpred = post["dpred"]
        return post

    def _sensitivities(self, x):
        """J (explicit) or a linearization at x, cached for the step."""
        import torch

        s = self._sens
        if s is not None and s["m"] is x:
            return s
        m_t = torch.as_tensor(x, dtype=self._dt, device=self._mdev)
        nD, nM = int(self._dobs_t.numel()), int(m_t.numel())
        self._mode = mode = self._choose_mode(nD, nM)
        self._sens = None                       # free the previous one first
        if mode == "explicit":
            if hasattr(self.engine, "linearize"):
                # rows streamed from one linearization: peak = J + fields
                lin = self.engine.linearize(m_t)
                J = lin.Jmatrix()
                del lin
            else:
                try:
                    J = self.engine.Jmatrix(m_t, block=self.jmatrix_block)
                except TypeError:                # an engine without the knob
                    J = self.engine.Jmatrix(m_t)
            J = J.to(device=self._mdev, dtype=self._dt)
            self.counts["jacobians"] += 1
            s = {"m": x, "mode": mode, "J": J,
                 "Jv": lambda v: J @ v, "Jtv": lambda w: J.T @ w}
        else:
            lin = self.engine.linearize(m_t)
            self.counts["linearizations"] += 1
            s = {"m": x, "mode": mode, "lin": lin,
                 "Jv": lin.Jvec, "Jtv": lin.Jtvec}
        self._sens = s
        return s

    def _residual_weights(self, d):
        """(W dphi/dr, D): the gradient weight and the Gauss-Newton weight."""
        r = self._residual(d)
        w = self._lin_weight(d)                 # dr/dd: W, or W_log / d
        if self._nu > 0:
            dphi = 2.0 * (self._nu + 1.0) * r / (self._nu + r ** 2)
            D = 2.0 * (self._nu + 1.0) / (self._nu + r ** 2) * w ** 2
        else:
            dphi = 2.0 * r
            D = 2.0 * w ** 2
        return w * dphi, D

    def _fill_gradient(self, post):
        import torch

        phi_m_t = self._build_phi_m_t()
        beta = float(self.parent.beta)
        s = self._sensitivities(post["m"])
        d = torch.as_tensor(post["dpred"], dtype=self._dt, device=self._mdev)
        wd, _ = self._residual_weights(d)
        g_d = s["Jtv"](wd)
        m2 = torch.as_tensor(post["m"], dtype=self._dt,
                             device=self._mdev).clone().requires_grad_(True)
        g_m, = torch.autograd.grad(phi_m_t(m2), m2)
        post["g_d"] = g_d.cpu().numpy()
        post["g_m"] = g_m.detach().cpu().numpy()
        post["g"] = post["g_d"] + beta * post["g_m"]

    # ------------------------------------------------------------------ #
    #  the regularization Hessian
    # ------------------------------------------------------------------ #
    def _phi_m_hessian(self):
        """H_m v and diag(H_m) of the extracted regularization (current W).

        H_m is assembled once per step into ONE sparse matrix, so a CG
        iteration costs one sparse product for it instead of two per
        regularization term."""
        import torch

        H = sp.csr_matrix(self._phi_m_hessian_scipy())
        H.sort_indices()
        dev, dt = self._mdev, self._dt
        Ht = torch.sparse_csr_tensor(
            torch.as_tensor(H.indptr, dtype=torch.int64, device=dev),
            torch.as_tensor(H.indices, dtype=torch.int64, device=dev),
            torch.as_tensor(H.data, dtype=dt, device=dev), H.shape)
        diag = torch.as_tensor(H.diagonal(), dtype=dt, device=dev)

        def Hm(v):
            return Ht @ v

        return Hm, diag

    def _phi_m_hessian_scipy(self):
        """H_m as a scipy CSC matrix, for the sparse factorization."""
        H = None
        for mult, (_, A_t, _, w_t) in zip(self._phi_m_mults, self._phi_m_terms):
            A = self._phi_m_A_np[id(A_t)]
            w2 = (w_t ** 2).cpu().numpy()
            part = (2.0 * mult) * (A.T @ sp.diags(w2) @ A)
            H = part if H is None else H + part
        return sp.csc_matrix(H)

    def _choose_mode(self, nD, nM):
        if self.jacobian in ("explicit", "matrix_free"):
            return self.jacobian
        if self.jacobian != "auto":
            raise ValueError(f"jacobian must be explicit|matrix_free|auto, got {self.jacobian!r}")
        explicit_ok = hasattr(self.engine, "Jmatrix") and \
            nD * nM * 8 <= self.explicit_max_gib * 2 ** 30
        if explicit_ok:
            return "explicit"
        if hasattr(self.engine, "linearize"):
            return "matrix_free"
        if hasattr(self.engine, "Jmatrix"):
            return "explicit"
        raise NotImplementedError(
            f"{type(self.engine).__name__} offers neither Jmatrix nor "
            "linearize; Gauss-Newton needs one of them")

    # ------------------------------------------------------------------ #
    #  the two inner solves
    # ------------------------------------------------------------------ #
    def _solve_direct(self, J, D, g, beta):
        """Exact step in data space (Woodbury); None if H_m is not definite."""
        import scipy.sparse.linalg as spla
        import torch

        try:
            lu = spla.splu(beta * self._phi_m_hessian_scipy())
        except RuntimeError:                     # singular: alpha_s = 0
            return None
        dev, dt = self._mdev, self._dt
        rhs = np.column_stack([J.T.cpu().numpy(), g.cpu().numpy()])
        sol = torch.as_tensor(lu.solve(rhs), dtype=dt, device=dev)
        if not torch.isfinite(sol).all():
            return None
        K, u = sol[:, :-1], sol[:, -1]
        S = J @ K
        S.diagonal().add_(1.0 / D)
        try:
            Ls = torch.linalg.cholesky(S)
        except RuntimeError:
            return None
        z = torch.cholesky_solve((J @ u).unsqueeze(1), Ls).squeeze(1)
        return -(u - K @ z)

    def _hm_solve_gpu(self, Hm, rhs_t):
        """H_m^-1 rhs on the gpu with cuDSS (H_m is SPD for alpha_s > 0, and
        its pattern does not change between steps: plan once, refactorize per
        step). Returns None on cpu, or if the residual says the solve is off --
        the caller then falls back to SuperLU on the host."""
        import torch

        if torch.device(self._mdev).type != "cuda":
            return None
        try:
            from .solver import make_solver
            H = sp.csr_matrix(Hm)
            H.sort_indices()
            n, k = H.shape[0], int(rhs_t.shape[1])
            c = getattr(self, "_hm_cache", None)
            if (c is None or c["k"] != k or c["n"] != n
                    or not np.array_equal(c["indptr"], H.indptr)
                    or not np.array_equal(c["indices"], H.indices)):
                crow = torch.as_tensor(H.indptr, dtype=torch.int32, device=self._mdev)
                col = torch.as_tensor(H.indices, dtype=torch.int32, device=self._mdev)
                c = {"k": k, "n": n, "indptr": H.indptr.copy(), "indices": H.indices.copy(),
                     "slv": make_solver(crow, col, n, k, device=self._mdev,
                                        dtype=torch.float64, matrix_type="spd"),
                     "crow64": crow.long(), "col64": col.long()}
                self._hm_cache = c
            vals = torch.as_tensor(H.data, dtype=torch.float64, device=self._mdev)
            c["slv"].factorize(vals)
            X = c["slv"].solve(rhs_t.to(torch.float64).contiguous())
            Ht = torch.sparse_csr_tensor(c["crow64"], c["col64"], vals, (n, n))
            res = torch.linalg.norm(Ht @ X - rhs_t) / torch.linalg.norm(rhs_t)
            if not torch.isfinite(res) or float(res) > 1e-8:
                return None
            return X.to(self._dt)
        except Exception:                        # any cuDSS / nvmath trouble
            return None

    def _occam(self, J, D, g_d, g_m, r, beta_now, nD, w=None):
        """Choose beta inside the step (Occam's inversion, Constable et al.
        1987), the way RES2DINV's 'combined Marquardt and Occam' does.

        With J fixed, the exact step for ANY beta is cheap in data space:
        factor H_m once, K0 = H_m^-1 J^T, u_d = H_m^-1 g_d, u_m = H_m^-1 g_m,
        and then for each beta

            u = u_d / beta + u_m,   S = D^-1 + J K0 / beta,
            p(beta) = -(u - K0 S^-1 J u / beta)

        -- no PDE solve, no CG; and with one eigendecomposition of
        D^1/2 J K0 D^1/2 per step (see the body) each candidate beta costs two
        nD x nD products in data space. The linearized misfit
        phi_d(r + W J p(beta)) predicts where each beta lands; the LARGEST
        beta whose prediction reaches the target (the smoothest model that
        fits) is taken, refined by bisection in log beta. When no candidate
        reaches the target the one that predicts the lowest misfit is taken,
        within `beta_span` of the current beta so a single step cannot drop it
        arbitrarily.

        Returns (beta, p, predicted phi_d), or None when H_m is singular
        (alpha_s = 0) and the search cannot run.
        """
        import scipy.sparse.linalg as spla
        import torch

        dev, dt = self._mdev, self._dt
        Hm = self._phi_m_hessian_scipy()
        rhs_t = torch.cat([J.T, g_d.unsqueeze(1), g_m.unsqueeze(1)], dim=1)
        sol = self._hm_solve_gpu(Hm, rhs_t)
        if sol is None:                          # cpu, or the gpu solve was off
            try:
                lu = spla.splu(Hm)
            except RuntimeError:
                return None
            sol = torch.as_tensor(lu.solve(rhs_t.cpu().numpy()), dtype=dt, device=dev)
        if not torch.isfinite(sol).all():
            return None
        K0, ud, um = sol[:, :-2], sol[:, -2], sol[:, -1]
        JK0, Jud, Jum = J @ K0, J @ ud, J @ um
        # ONE eigendecomposition serves every beta: with Dh = D^1/2,
        #   S(beta) = D^-1 + J K0 / beta = Dh^-1 (I + M / beta) Dh^-1,
        #   M = Dh (J K0) Dh = Q diag(lam) Q^T   (J K0 = J H_m^-1 J^T, sym. PSD)
        # so z(beta) = S^-1 J u(beta) = Dh Q (f * c),  f = 1 / (1 + lam / beta),
        # c(beta) = Q^T Dh J u(beta) = cd / beta + cm, and the predicted data
        # change J p(beta) = -(J u - Dh^-1 Q ((lam / beta) f * c)) costs two
        # nD x nD products per beta -- no Cholesky, nothing of size nM.
        Dh = torch.sqrt(D)
        Msym = Dh.unsqueeze(1) * (0.5 * (JK0 + JK0.T)) * Dh.unsqueeze(0)
        lam, Q = torch.linalg.eigh(Msym)
        lam = torch.clamp(lam, min=0.0)
        cd, cm = Q.T @ (Dh * Jud), Q.T @ (Dh * Jum)

        def coef(b):
            return cd / b + cm, 1.0 / (1.0 + lam / b)

        def Jp_of(b):
            c, f = coef(b)
            Ju = Jud / b + Jum
            return -(Ju - (Q @ ((lam / b) * f * c)) / Dh)

        def step(b):
            c, f = coef(b)
            z = Dh * (Q @ (f * c))
            return -((ud / b + um) - (K0 @ z) / b)

        def predict(b):
            return float(self._phi_d_of_r(r + w_lin * Jp_of(b)))

        w_lin = self._W_t if w is None else w
        T = self.beta_aim * self.chi_target * nD
        hi, lo = self.beta_span
        betas = beta_now * np.logspace(np.log10(hi), np.log10(lo), self.beta_points)
        vals = [predict(b) for b in betas]                  # betas descending
        reach = [i for i, v in enumerate(vals) if v <= T]
        if reach and reach[0] > 0:
            a_, c_ = np.log(betas[reach[0] - 1]), np.log(betas[reach[0]])
            for _ in range(12):                             # a_ misses, c_ reaches
                mid = 0.5 * (a_ + c_)
                if predict(np.exp(mid)) <= T:
                    c_ = mid
                else:
                    a_ = mid
            b_c = float(np.exp(c_))
        elif reach:
            b_c = float(betas[0])
        else:
            b_c = float(betas[int(np.argmin(vals))])
        p = step(b_c)
        if not torch.isfinite(p).all():
            return None
        return b_c, p, predict(b_c)

    @timeIt
    def findSearchDirection(self):
        import torch

        self._torch_setup()
        phi_m_t = self._build_phi_m_t()
        invProb = self.parent
        beta = float(invProb.beta)
        dev, dt = self._mdev, self._dt
        x = self.xc

        post = self._post
        if post is None or post["m"] is not x:
            post = self._eval_point(x)
        if post.get("g") is None:
            self._fill_gradient(post)
        s = self._sensitivities(post["m"])

        m_t = torch.as_tensor(post["m"], dtype=dt, device=dev)
        d = torch.as_tensor(post["dpred"], dtype=dt, device=dev)
        _, D = self._residual_weights(d)
        g = torch.as_tensor(post["g_d"], dtype=dt, device=dev) \
            + beta * torch.as_tensor(post["g_m"], dtype=dt, device=dev)
        f0 = post["phi_d"] + beta * post["phi_m"]
        nD = int(D.numel())

        free = torch.ones_like(m_t)
        if self.lower is not None:
            free[(m_t <= float(self.lower)) & (g > 0)] = 0.0
        if self.upper is not None:
            free[(m_t >= float(self.upper)) & (g < 0)] = 0.0
        pinned = bool((free == 0).any())

        p = None
        if self.beta_search:
            if s["mode"] != "explicit" or pinned:
                if not getattr(self, "_warned_occam", False):
                    import warnings
                    warnings.warn("beta_search needs the explicit J and no cell "
                                  "pinned on a bound; this step uses the fixed "
                                  "beta instead.", stacklevel=2)
                    self._warned_occam = True
            else:
                g_d_t = torch.as_tensor(post["g_d"], dtype=dt, device=dev)
                g_m_t = torch.as_tensor(post["g_m"], dtype=dt, device=dev)
                r = self._residual(d)
                occ = self._occam(s["J"], D, g_d_t, g_m_t, r, beta, nD,
                                  w=self._lin_weight(d))
                if occ is not None:
                    beta, p, self._phi_pred = occ
                    invProb.beta = beta          # the objective of this step
                    g = g_d_t + beta * g_m_t
                    f0 = post["phi_d"] + beta * post["phi_m"]
                    self.counts["occam"] += 1
                    self._inner_tag = "occam"

        want_direct = (self.inner == "direct" or
                       (self.inner == "auto" and nD <= self.direct_max_data))
        if p is None and want_direct and s["mode"] == "explicit" and not pinned:
            p = self._solve_direct(s["J"], D, g, beta)
            if p is not None:
                self.counts["direct"] += 1
                self._inner_tag = "direct"
        if p is None:
            Hm, Hm_diag = self._phi_m_hessian()
            Jv, Jtv = s["Jv"], s["Jtv"]

            def H(v):
                v = free * v
                return free * (Jtv(D * Jv(v)) + beta * Hm(v))

            if self.precond == "jacobi":
                if s["mode"] == "explicit":
                    dd = (D.unsqueeze(1) * s["J"] ** 2).sum(0)
                else:
                    dd = s["lin"].diag_JtDJ(D)
                diag = dd + beta * Hm_diag
                Minv = free / torch.clamp(diag, min=1e-12 * float(diag.max()))
            else:
                Minv = free.clone()
            p, it = _pcg(H, -free * g, Minv, self.cg_maxiter, self.cg_rtol)
            self.counts["cg"] += it
            self._inner_tag = f"cg{it}"
        self._sens = None                        # J / fields are not reused

        if self.max_step is not None:
            big = float(p.abs().max())
            if big > self.max_step:
                p = p * (self.max_step / big)
        slope = float(g @ p)
        if not slope < 0.0:                      # cannot happen for SPD H; guard
            p = -g
            slope = float(g @ p)
        p_np = p.detach().cpu().numpy()

        # ---- line search: Armijo, backtracking along the parabola ----------
        def trial(xt):
            with torch.no_grad():
                mt = torch.as_tensor(xt, dtype=dt, device=dev)
                fd, dt_pred = self._phi_d_t(mt)
                return float(fd) + beta * float(phi_m_t(mt)), dt_pred

        xc = np.asarray(x, dtype=float)
        a, best = 1.0, (f0, 0.0, None)
        self._ls_ok = False
        for _ in range(self.ls_maxiter):
            ft, dpt = trial(self.projection(xc + a * p_np))
            self.counts["ls_forwards"] += 1
            if ft < best[0]:
                best = (ft, a, dpt)
            if ft <= f0 + self.ls_c1 * a * slope:
                self._ls_ok = True
                break
            curv = ft - f0 - slope * a
            a_q = -slope * a * a / (2.0 * curv) if curv > 0 else 0.5 * a
            a = min(max(a_q, 0.1 * a), 0.5 * a)
        _, a, d_best = best
        self._ls_step = a
        if a > 0.0:
            self._eval_point(self.projection(xc + a * p_np), d=d_best)
        # else: no decrease found; stay, the post at x keeps its gradient
        return self._post["m"] - self.xc

    @timeIt
    def modifySearchDirection(self, p):
        xt, _ = _TorchMinimize.modifySearchDirection(self, p)
        return xt, self._ls_step > 0.0
