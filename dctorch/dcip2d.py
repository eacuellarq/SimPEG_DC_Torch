"""
JOINT NON-LINEAR DC + IP 2.5D forward on GPU (Seigel, unlinearized).

Step 2 of the IP ladder. Where `TorchIP2D` linearizes chargeability around a
FIXED background (stock SimPEG's only option), this module solves the actual
two-state problem and lets autograd differentiate both fields exactly:

    sigma      = exp(-m_rho)                     (unpolarized / instantaneous)
    eta        = eta_max * sigmoid(m_eta)        (bounds (0, eta_max) IN GRAPH)
    sigma_eff  = sigma * (1 - eta)               (fully polarized / steady DC)

    V0 = F(sigma)        (instantaneous / high-frequency voltage, V_inf)
    V1 = F(sigma_eff)    (fully polarized steady-state voltage,   V_tot)

    d_IP = ip_scale * (V1 - V0) / V1                            [V/V or mV/V]
    d_DC = V1  (dc_state="total") | V0  (dc_state="instantaneous")   [volt]

`d_IP` is the EXACT Seigel apparent chargeability eta_a = (V_tot - V_inf)/V_tot.
Stock's linearization puts the background voltage in the denominator and drops
the O(eta^2) term of the numerator, so the two agree to first order — that
difference IS the consistency gate of this module (`error = O(eta)`), since
SimPEG has no non-linear IP simulation to compare against. On a uniform
chargeable halfspace the gap is exact and instructive: the true apparent
chargeability is eta, while the linearized forward predicts eta/(1-eta), so a
linearized inversion of true data lands on eta/(1+eta) — a 29% underestimate
at eta = 0.4.

`dc_state` is the OTHER modelling choice the linearized workflow hides. The
voltage a time-domain DC/IP receiver reports as "the resistivity datum" is the
on-time steady state, i.e. the FULLY POLARIZED V_tot = F(sigma(1-eta)) — so
"total" (the default) is the faithful convention for field data, and it makes
the two datasets genuinely coupled: DC fixes sigma(1-eta), IP fixes the split.
The sequential workflow instead feeds the DC-recovered model into the
linearized IP as if it were the instantaneous sigma, an O(eta) inconsistency
on top of the linearization. "instantaneous" reproduces that convention (and
is what the parity gates use to check the sigma channel against stock DC).

Cost: the two states share mesh, survey and SPARSITY PATTERN, so a joint
forward is ONE `CuDSSBatchSolver` sequence-batch of 2*nky systems (values
differ, plan/symbolic factorization shared) — the same machinery that already
batches the wavenumbers, just twice as wide. The backward is the batched
adjoint, so the gradient w.r.t. BOTH fields comes out of a single vjp with no
extra solves beyond the ones the adjoint method needs anyway.

Model convention: `m = cat(m_rho, m_eta)` (each of length nP = active cells,
or nC without an active mask). m_rho = log(rho) as everywhere else in dctorch;
m_eta = logit(eta / eta_max), so eta stays strictly inside (0, eta_max) with
NO clamping and the regularization acts on the unbounded variable.

Data layout: the engine holds ONE survey (all receivers `data_type="volt"` —
the 1/V normalization is done here, explicitly, not by the receiver) and two
index arrays into its data vector: `idx_dc` selects the DC data, `idx_ip` the
IP data. When both datasets share the geometry (the usual field case) both are
`arange(nD)`.
"""
import numpy as np
import torch

from .solver import make_batch_solver, sparse_solve_batch


class TorchDCIP2D:
    """Differentiable JOINT (rho, eta) 2.5D nodal DC-IP forward.

    Parameters
    ----------
    mesh, survey : the joint survey; receivers must be plain "volt" (this
        class forms the chargeability ratio itself).
    nky : wavenumbers (the batch is 2*nky systems).
    active_cells, val_inactive : as in `TorchDC2D`; the model (both blocks)
        lives on the active cells, inactive eta = 0.
    idx_dc, idx_ip : int arrays into the survey data ordering (default: all).
    eta_max : upper bound of the sigmoid (1.0 = the physical bound).
    ip_scale : multiplies d_IP (1000 for data in mV/V).
    dc_state : "total" (default; the DC datum is the fully polarized
        steady-state voltage a time-domain receiver measures) or
        "instantaneous" (the un-polarized voltage — the convention the
        linearized workflow implicitly assumes). See the module docstring.
    """

    def __init__(self, mesh, survey, nky=12, device="cuda",
                 dtype=torch.float64, active_cells=None, val_inactive=None,
                 idx_dc=None, idx_ip=None, eta_max=1.0, ip_scale=1.0,
                 dc_state="total", quadrature=None):
        from .simulation2d import TorchDC2D

        dc = TorchDC2D(mesh, survey, nky=nky, device=device, dtype=dtype,
                       active_cells=active_cells, val_inactive=val_inactive,
                       quadrature=quadrature)
        self.dc = dc
        self.device, self.dtype = dc.device, dc.dtype
        self.nky, self.nD = dc.nky, dc.nD
        self.nP = dc.nC if dc._P_inj is None else dc._P_inj.shape[1]
        self.eta_max = float(eta_max)
        self.ip_scale = float(ip_scale)
        assert dc_state in ("total", "instantaneous"), dc_state
        self.dc_state = dc_state

        # the joint forward needs 2*nky systems on the SAME pattern: swap the
        # DC engine's nky-wide batch solver for a 2*nky one (the plan is
        # per-pattern, so this only re-runs the symbolic stage once at build)
        crow, col = dc.solver.crow, dc.solver.col
        dc.solver.free()
        dc.solver = None
        self.solver = make_batch_solver(crow, col, dc.nN, dc.nS, 2 * dc.nky,
                                        device=self.device, dtype=self.dtype)

        def _idx(a):
            if a is None:
                return torch.arange(dc.nD, dtype=torch.int64,
                                    device=self.device)
            return torch.as_tensor(np.asarray(a, dtype=np.int64),
                                   device=self.device)

        self.idx_dc, self.idx_ip = _idx(idx_dc), _idx(idx_ip)
        self.nD_dc = int(self.idx_dc.numel())
        self.nD_ip = int(self.idx_ip.numel())

    # ---------------- model <-> fields (all inside the graph) --------------
    def _inject_rho(self, m_rho):
        if self.dc._P_inj is None:
            return m_rho
        return torch.sparse.mm(self.dc._P_inj,
                               m_rho.unsqueeze(1)).squeeze(1) + self.dc._v_inj

    def _inject_eta(self, eta):
        if self.dc._P_inj is None:
            return eta
        return torch.sparse.mm(self.dc._P_inj, eta.unsqueeze(1)).squeeze(1)

    def split(self, m):
        """(m_rho, m_eta) blocks of the stacked model."""
        m = m.reshape(-1)
        assert m.numel() == 2 * self.nP, \
            f"model must be 2*{self.nP} long, got {m.numel()}"
        return m[:self.nP], m[self.nP:]

    def eta_of(self, m_eta):
        """Physical eta (on the model's cells) from the logit block."""
        return self.eta_max * torch.sigmoid(m_eta)

    def m_eta_of(self, eta):
        """Inverse map (numpy or tensor), for reference models / starts."""
        if not torch.is_tensor(eta):
            e = np.clip(np.asarray(eta, dtype=float) / self.eta_max,
                        1e-12, 1 - 1e-12)
            return np.log(e / (1 - e))
        e = (eta / self.eta_max).clamp(1e-12, 1 - 1e-12)
        return torch.log(e / (1 - e))

    # ---------------- runtime (torch, differentiable) ----------------------
    def _forward(self, sig_full, eta_full):
        """Both voltage vectors (nD,) from full-mesh sigma and eta."""
        dc = self.dc
        sig_eff = sig_full * (1.0 - eta_full)
        # one sparse matmul for both states, then the 2*nky value stack
        two = torch.stack((sig_full, sig_eff), dim=1)          # (nC, 2)
        V = torch.sparse.mm(dc.Lstack, two)                    # (nky*nnzA, 2)
        VALS = torch.cat((V[:, 0].view(dc.nky, dc.nnzA),
                          V[:, 1].view(dc.nky, dc.nnzA)), dim=0)
        U = sparse_solve_batch(VALS, dc.q, self.solver)        # (2nky, nN, nS)
        w = dc.wq.view(-1, 1, 1)
        Ubar = torch.stack(((w * U[:dc.nky]).sum(0),
                            (w * U[dc.nky:]).sum(0)), dim=0)   # (2, nN, nS)
        d2 = torch.stack([torch.sparse.mm(dc.Pd, Ubar[i])[dc.iD, dc.sidx]
                          for i in range(2)], dim=0)           # (2, nD)
        return d2[0], d2[1]

    def dpred_from_fields(self, m_rho, eta):
        """(d_DC, d_IP) from log-rho and PHYSICAL eta (both on the model's
        cells). Bypasses the sigmoid — used by the consistency gates."""
        dev0, dt0 = m_rho.device, m_rho.dtype
        mr = m_rho.to(device=self.device, dtype=self.dtype).reshape(-1)
        e = eta.to(device=self.device, dtype=self.dtype).reshape(-1)
        sig = torch.exp(-self._inject_rho(mr))
        V0, V1 = self._forward(sig, self._inject_eta(e))
        d_dc = (V1 if self.dc_state == "total" else V0)[self.idx_dc]
        d_ip = self.ip_scale * (V1[self.idx_ip] - V0[self.idx_ip]) \
            / V1[self.idx_ip]
        return d_dc.to(device=dev0, dtype=dt0), d_ip.to(device=dev0, dtype=dt0)

    def dpred(self, m):
        """(d_DC, d_IP) for the stacked model m = cat(log-rho, logit-eta)."""
        m_rho, m_eta = self.split(m)
        return self.dpred_from_fields(m_rho, self.eta_of(m_eta))

    def misfit(self, m, dobs_dc, w_dc, dobs_ip, w_ip, lam=1.0):
        """phi_d^DC + lam * phi_d^IP, each ||w*(dpred-dobs)||^2 (no 1/2 —
        the 0.25 convention; the drivers add their own 1/2)."""
        d_dc, d_ip = self.dpred(m)
        r0 = w_dc * (d_dc - dobs_dc)
        r1 = w_ip * (d_ip - dobs_ip)
        return (r0 ** 2).sum() + lam * (r1 ** 2).sum()
