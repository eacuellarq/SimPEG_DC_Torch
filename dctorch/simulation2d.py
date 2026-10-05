"""
Differentiable 2.5D DC-resistivity forward on GPU, driven by STOCK SimPEG 0.25.

`TorchDC2D` wraps a stock `Simulation2DNodal` problem definition (mesh, survey,
nky, Robin BC). At build time it runs SimPEG's own numpy assembly ONCE to
extract, per wavenumber ky, a sparse LINEAR map from conductivity to the CSR
values of the system matrix (assembly is linear in sigma):

    A(ky).data = L_ky @ sigma,   L_ky = T1 @ Qe + ky^2 * Td @ Qn + Td @ AvgBC[ky]

(T1 = Grad outer-product scatter — general row expansion, TreeMesh hanging
nodes included; Qe = edge inner-product deriv; Qn = aveN2CC.T * vol; Td =
diagonal scatter.) Every map is validated against `sim.getA(ky)` at two models
before use. At runtime a forward is: one sparse matmul (stacked L), one cuDSS
SEQUENCE-batch solve over the nky systems (shared pattern, no block-diagonal
super-matrix), one stacked receiver projection — fully autograd-traceable, so
J^T v is an exact vjp.

Model convention: m = log(rho) on the full mesh (sigma = exp(-m)), matching a
stock sim built with rhoMap=maps.ExpMap(mesh).

Ported from the 0.18-fork engine (solver/batched_eval.py + _dpred_torch_fast),
now living OUTSIDE simpeg: nothing here patches site-packages, and simpeg /
discretize never see a torch tensor.
"""
import numpy as np
import scipy.sparse as sp
import torch

from .solver import (make_batch_solver, sparse_solve_batch,
                     sparse_solve_batch_prefactored)


class TorchDC2D:
    """Differentiable dpred/misfit for 2.5D nodal DC on a fixed mesh+survey."""

    def __init__(self, mesh, survey, nky=12, device="cuda",
                 dtype=torch.float64, active_cells=None,
                 val_inactive=None, chunk=None, singularity_removal=False,
                 rho_primary=None, quadrature=None):
        """quadrature : how the nky wavenumbers and weights are chosen.

        - "electrodes" (the default): fitted over the survey's electrode
          distances with non-negative weights, as RES2DINV fits its own
          (`dctorch.quadrature.k0_quadrature`). With the default nky = 12, on
          a 661-measurement field line the forward error against a 31-point
          reference is 0.0007 %
          median, against 0.035 % for SimPEG's own 12 points; the inverted
          model is the same. Points the fit gives zero weight are dropped, so
          `self.nky` can come out below `nky`. Fewer points are not
          equivalent on a real mesh (7 reach 2.5 % forward error): keep 12
          unless a reference on your mesh says otherwise.
        - "simpeg": stock SimPEG's, so dpred matches a stock simulation bit for
          bit (a fallback to its trapezoid rule is warned about).
        - "auto": SimPEG's unless its fit failed and fell back to the trapezoid
          rule (it does at nky = 31, ~2.5 % error); then the electrode fit.
        - None: the environment variable DCTORCH_QUADRATURE if set (e.g.
          "simpeg", to reproduce results made before the default changed),
          else "electrodes".

        `self.quadrature_info` records the choice, the transform error on the
        electrode-distance range, the wavenumbers used and the error
        amplification sum|w|/sum w (1 for positive weights).
        """
        import warnings
        from simpeg import maps
        from simpeg.electromagnetics.static import resistivity as dc
        from .quadrature import (amplification, electrode_distance_range,
                                 k0_quadrature, transform_error)

        if quadrature is None:
            import os
            quadrature = os.environ.get("DCTORCH_QUADRATURE", "electrodes")
        if quadrature not in ("auto", "simpeg", "electrodes"):
            raise ValueError(f"quadrature must be auto|simpeg|electrodes, got {quadrature!r}")
        self.device = torch.device(device)
        self.dtype = dtype
        self.chunk = None if chunk is None else int(chunk)
        self._sr = bool(singularity_removal)
        self._rho_p = None if rho_primary is None else float(rho_primary)
        self._primary = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ext = dc.simulation_2d.Simulation2DNodal(
                mesh, survey=survey, rhoMap=maps.ExpMap(mesh), nky=nky)
        fell_back = False
        for w in caught:
            if "trapezoidal" in str(w.message).lower():
                fell_back = True
            else:                                   # not ours to swallow
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

        rmin, rmax = electrode_distance_range(survey)
        err_simpeg = transform_error(ext._quad_points, ext._quad_weights, rmin, rmax)
        self.quadrature_info = {"source": "simpeg", "simpeg_fell_back": fell_back,
                                "r_range": (rmin, rmax), "max_rel_err": err_simpeg,
                                "amplification": amplification(ext._quad_weights),
                                "n_used": int(np.size(ext._quad_points))}
        if quadrature == "electrodes" or (quadrature == "auto" and fell_back):
            pts, wts, info = k0_quadrature(rmin, rmax, nky)
            ext._quad_points, ext._quad_weights = pts, wts
            self.quadrature_info.update(source="electrodes",
                                        max_rel_err=info["max_rel_err"],
                                        amplification=info["amplification"],
                                        n_used=info["n_used"],
                                        simpeg_max_rel_err=err_simpeg)
            if fell_back:
                warnings.warn(
                    f"SimPEG's wavenumber fit failed at nky={nky} and fell back to "
                    f"a trapezoid rule (transform error {err_simpeg:.1e} on the "
                    f"electrode distances); replaced by the electrode fit "
                    f"({info['max_rel_err']:.1e}).", stacklevel=2)
        elif fell_back:
            warnings.warn(
                f"SimPEG's wavenumber fit failed at nky={nky}; its trapezoid "
                f"fallback has transform error {err_simpeg:.1e} on the electrode "
                f"distances. Use quadrature='electrodes' or another nky.",
                stacklevel=2)
        self._extract(ext, mesh, survey)
        # active-cells convention (standard protocol): the model lives on the
        # active cells; inactive (air) cells are fixed at val_inactive log-rho
        self._P_inj = None
        if active_cells is not None:
            act = np.asarray(active_cells, dtype=bool)
            ia = np.where(act)[0]
            self._P_inj = torch.sparse_coo_tensor(
                torch.tensor(np.vstack((ia, np.arange(ia.size))),
                             dtype=torch.int64, device=self.device),
                torch.ones(ia.size, dtype=self.dtype, device=self.device),
                (mesh.nC, ia.size)).coalesce()
            v = np.full(mesh.nC, float(val_inactive))
            v[act] = 0.0
            self._v_inj = torch.tensor(v, dtype=self.dtype,
                                       device=self.device)
        self._mesh, self._survey = mesh, survey
        self._PdT = None        # dense Pd^T, the adjoint rhs of every datum
        self._PdTs = None       # the same, sparse (Linearization2D.Jtvec)
        self._PinjT = None

    # ---------------- build-time extraction (numpy, stock SimPEG) ----------
    def _extract(self, ext, mesh, survey):
        nC, nN = mesh.nC, mesh.n_nodes
        m0 = np.log(100.0) * np.ones(nC)
        ext.model = m0
        sig0 = np.exp(-m0)
        kys = np.asarray(ext._quad_points, dtype=float)
        wq = np.asarray(ext._quad_weights, dtype=float)
        nky = len(kys)

        A_refs = []
        for ky in kys:                       # also populates _AvgBC via setBC
            A = sp.csr_matrix(ext.getA(ky))
            A.sort_indices()
            A_refs.append(A)
        A0 = A_refs[0]
        for A in A_refs[1:]:
            assert np.array_equal(A0.indptr, A.indptr) and \
                np.array_equal(A0.indices, A.indices), \
                "A sparsity pattern differs across ky"
        Ac = A0.tocoo()
        rows_c = Ac.row.astype(np.int64)
        cols_c = Ac.col.astype(np.int64)
        nnzA = A0.nnz
        keys = rows_c * nN + cols_c
        assert np.all(np.diff(keys) > 0)

        def pos_of(r, c):
            k = r.astype(np.int64) * nN + c.astype(np.int64)
            p = np.searchsorted(keys, k)
            assert np.all(keys[p] == k), "entry missing from A pattern"
            return p

        # Grad term as scatter of edge weights: A1(w) = Grad.T diag(w) Grad.
        # Rows grouped by nnz count so TreeMesh hanging-node rows (>2 entries)
        # expand all index pairs.
        Grad = sp.csr_matrix(mesh.nodal_gradient)
        nE = Grad.shape[0]
        cnt = np.diff(Grad.indptr)
        rA_l, cA_l, cf_l, eI_l = [], [], [], []
        for k in np.unique(cnt):
            rows_k = np.where(cnt == k)[0]
            gather = Grad.indptr[rows_k][:, None] + np.arange(k)[None, :]
            I = Grad.indices[gather]
            V = Grad.data[gather]
            a = np.repeat(np.arange(k), k)
            b = np.tile(np.arange(k), k)
            rA_l.append(I[:, a].ravel())
            cA_l.append(I[:, b].ravel())
            cf_l.append((V[:, a] * V[:, b]).ravel())
            eI_l.append(np.repeat(rows_k, k * k))
        T1 = sp.coo_matrix(
            (np.concatenate(cf_l),
             (pos_of(np.concatenate(rA_l), np.concatenate(cA_l)),
              np.concatenate(eI_l))),
            shape=(nnzA, nE)).tocsr()

        Me_test = sp.csr_matrix(mesh.get_edge_inner_product(model=sig0))
        assert (Me_test - sp.diags(Me_test.diagonal())).nnz == 0, \
            "MeSigma not diagonal"
        Qe = sp.csr_matrix(
            mesh.get_edge_inner_product_deriv(np.ones(nC))(np.ones(nE)))
        assert np.allclose(Qe @ sig0, Me_test.diagonal(), rtol=1e-12)

        Qn = sp.csr_matrix(mesh.aveN2CC.T) @ sp.diags(mesh.cell_volumes)
        Td = sp.coo_matrix(
            (np.ones(nN), (pos_of(np.arange(nN), np.arange(nN)),
                           np.arange(nN))), shape=(nnzA, nN)).tocsr()
        L_edge = T1 @ Qe
        L_node = Td @ Qn

        # validate the linear maps at two models before trusting them
        rng = np.random.RandomState(7)
        m1 = m0 + 0.5 * rng.randn(nC)
        sig1 = np.exp(-m1)
        L_all = []
        for iky, ky in enumerate(kys):
            L = L_edge + (ky ** 2) * L_node
            if getattr(ext, "_AvgBC", None) is not None and ky in ext._AvgBC:
                L = L + Td @ sp.csr_matrix(ext._AvgBC[ky])
            e0 = np.abs(L @ sig0 - A_refs[iky].data).max() \
                / np.abs(A_refs[iky].data).max()
            ext.model = m1
            A1v = sp.csr_matrix(ext.getA(ky))
            A1v.sort_indices()
            e1 = np.abs(L @ sig1 - A1v.data).max() / np.abs(A1v.data).max()
            ext.model = m0
            assert max(e0, e1) < 1e-10, \
                f"L_ky validation failed at ky={ky}: {e0:.2e}/{e1:.2e}"
            L_all.append(sp.csr_matrix(L))

        q = np.asarray(ext.getRHS(kys[0]))
        assert np.allclose(q, np.asarray(ext.getRHS(kys[-1]))), \
            "RHS depends on ky"
        nS = q.shape[1]

        Pb, sidx = [], []
        for isrc, src in enumerate(survey.source_list):
            for rx in src.receiver_list:
                P = sp.csr_matrix(rx.getP(mesh, "N"))
                Pb.append(P)
                sidx += [isrc] * P.shape[0]
        Pd = sp.vstack(Pb).tocsr()
        nD = Pd.shape[0]
        assert nD == survey.nD

        # ---- to torch ----
        dev, dt = self.device, self.dtype

        def s2t(M):
            M = M.tocoo()
            return torch.sparse_coo_tensor(
                torch.tensor(np.vstack((M.row, M.col)), dtype=torch.int64,
                             device=dev),
                torch.tensor(M.data, dtype=dt, device=dev),
                M.shape).coalesce()

        self.kys = kys
        self.nC, self.nN, self.nS, self.nD, self.nky = nC, nN, nS, nD, nky
        self.nnzA = nnzA
        self.Lstack = s2t(sp.vstack(L_all))          # (nky*nnzA, nC)
        self.wq = torch.tensor(wq, dtype=dt, device=dev)
        self.q = torch.tensor(q, dtype=dt, device=dev)
        self.Pd = s2t(Pd)
        self.sidx = torch.tensor(np.asarray(sidx), dtype=torch.int64,
                                 device=dev)
        self.iD = torch.arange(nD, dtype=torch.int64, device=dev)
        self._crow = torch.tensor(A0.indptr, dtype=torch.int32)
        self._ccol = torch.tensor(A0.indices, dtype=torch.int32)
        self._solvers = {}
        self.solver = self._solver_for(nS)

    # ---------------- source batching --------------------------------------
    def _solver_for(self, n_rhs):
        """One solver per right-hand-side width; the symbolic phase is cached
        inside each, so a run needs at most two (the chunk and the remainder)."""
        if n_rhs not in self._solvers:
            self._solvers[n_rhs] = make_batch_solver(
                self._crow, self._ccol, self.nN, n_rhs, self.nky,
                device=self.device, dtype=self.dtype)
        return self._solvers[n_rhs]

    def _source_chunks(self, chunk=None):
        w = self.chunk if chunk is None else chunk
        if not w or w >= self.nS:
            return [(0, self.nS)]
        return [(s, min(s + w, self.nS)) for s in range(0, self.nS, w)]

    def _rows_of(self, s0, s1):
        """Data rows fed by sources [s0, s1), and their column inside the chunk."""
        key = (s0, s1)
        if key not in getattr(self, "_rowcache", {}):
            if not hasattr(self, "_rowcache"):
                self._rowcache = {}
            sel = (self.sidx >= s0) & (self.sidx < s1)
            self._rowcache[key] = (torch.nonzero(sel, as_tuple=True)[0],
                                   self.iD[sel], self.sidx[sel] - s0)
        return self._rowcache[key]

    def _model_to_full(self, m):
        m_loc = m.to(device=self.device, dtype=self.dtype).reshape(-1)
        if self._P_inj is not None:
            m_loc = torch.sparse.mm(
                self._P_inj, m_loc.unsqueeze(1)).squeeze(1) + self._v_inj
        return m_loc

    # ---------------- singularity removal ----------------------------------
    def _ensure_primary(self):
        """Build the geometric part of the primary once. It carries no
        resistivity: each source scales it by the background the current model
        holds at its own electrode, unless rho_primary fixed one."""
        if self._primary is None:
            from .singularity import Primary
            self._primary = Primary(self, self._mesh, self._survey, self._rho_p)
        return self._primary

    def _solve_chunk(self, VALS, s0, s1, width, solver, prefactored=False,
                     m_full=None):
        """Fields of sources [s0, s1) as (nky, nN, width), by whichever split is
        in force. Differentiable in VALS; with singularity removal the model
        enters the right-hand side and the background of the primary as well as
        the operator."""
        solve = (sparse_solve_batch_prefactored if prefactored
                 else sparse_solve_batch)
        if not self._sr:
            q = self.q[:, s0:s1]
            if width > s1 - s0:
                q = torch.cat([q, q.new_zeros(self.nN, width - (s1 - s0))], dim=1)
            return solve(VALS, q.contiguous(), solver)
        pri = self._primary
        u1 = pri.chunk(s0, s1, width)
        rho_s = pri.scale(m_full, s0, s1, width)
        rhs = pri.rhs(VALS, u1, rho_s)
        return u1 * rho_s + solve(VALS, rhs.contiguous(), solver)

    def _dpred_chunk(self, m_loc, s0, s1):
        """Data of the sources in [s0, s1), solved as one block."""
        sig = torch.exp(-m_loc).unsqueeze(1)
        VALS = torch.sparse.mm(self.Lstack, sig).view(self.nky, self.nnzA)
        w = s1 - s0
        U = self._solve_chunk(VALS, s0, s1, w, self._solver_for(w), m_full=m_loc)
        Ubar = (self.wq.view(-1, 1, 1) * U).sum(0)
        _, gi, li = self._rows_of(s0, s1)
        return torch.sparse.mm(self.Pd, Ubar)[gi, li]

    def misfit_and_grad(self, m, dobs, w, chunk=None):
        """phi_d = ||w (dpred - dobs)||^2 and its gradient, computed one source
        chunk at a time: the system is factorized once, each chunk is solved and
        differentiated against that factorization, and its fields are freed
        before the next chunk is formed. The whole-survey path holds every field
        of the survey at once, which is what bounds the survey a card can take.
        Returns (value, gradient) outside the autograd graph."""
        width = self.chunk if chunk is None else chunk
        width = self.nS if not width else min(int(width), self.nS)
        solver = self._solver_for(width)
        if self._sr:
            self._ensure_primary()
        with torch.no_grad():                      # one numeric factorization
            sig = torch.exp(-self._model_to_full(m)).unsqueeze(1)
            solver.factorize(torch.sparse.mm(self.Lstack, sig).view(self.nky, self.nnzA))
        total, grad = 0.0, torch.zeros_like(m)
        for s0, s1 in self._source_chunks(width):
            rows, gi, li = self._rows_of(s0, s1)
            mc = m.detach().clone().requires_grad_(True)
            m_full = self._model_to_full(mc)
            sig = torch.exp(-m_full).unsqueeze(1)
            VALS = torch.sparse.mm(self.Lstack, sig).view(self.nky, self.nnzA)
            U = self._solve_chunk(VALS, s0, s1, width, solver, prefactored=True,
                                  m_full=m_full)
            Ubar = (self.wq.view(-1, 1, 1) * U).sum(0)[:, :s1 - s0]
            d = torch.sparse.mm(self.Pd, Ubar)[gi, li]
            f = ((w[rows] * (d - dobs[rows])) ** 2).sum()
            f.backward()
            total += float(f)
            grad += mc.grad
        return total, grad

    # ---------------- runtime (torch, differentiable) -----------------------
    def dpred(self, m):
        """Predicted data (nD,) for m = log(rho) (full mesh, or active cells
        when built with active_cells). Autograd-traceable."""
        dev0, dt0 = m.device, m.dtype
        if self._sr:
            self._ensure_primary()
        m_loc = self._model_to_full(m)
        if self.chunk and self.chunk < self.nS:
            d = torch.empty(self.nD, device=self.device, dtype=self.dtype)
            for s0, s1 in self._source_chunks():
                rows, _, _ = self._rows_of(s0, s1)
                d[rows] = self._dpred_chunk(m_loc, s0, s1)
            return d.to(device=dev0, dtype=dt0)
        sig = torch.exp(-m_loc).unsqueeze(1)
        VALS = torch.sparse.mm(self.Lstack, sig).view(self.nky, self.nnzA)
        U = self._solve_chunk(VALS, 0, self.nS, self.nS, self.solver, m_full=m_loc)
        Ubar = (self.wq.view(-1, 1, 1) * U).sum(0)
        d = torch.sparse.mm(self.Pd, Ubar)[self.iD, self.sidx]
        return d.to(device=dev0, dtype=dt0)

    def Jtvec(self, m, v, chunk=None):
        """J^T v for J = d(dpred)/dm, without forming J.

        This is the same reverse pass the gradient uses, with an arbitrary
        vector in place of the misfit residual: one extra simulation for the
        whole product, and nothing of size nD x nM is ever allocated. The
        sources are walked in chunks exactly as in `misfit_and_grad`, so the
        fields of one chunk are freed before the next is formed and the
        factorization is shared by all of them.

        m : (nM,) log-resistivity.  v : (nD,) any data-space vector.
        Returns (nM,) outside the autograd graph, on m's device and dtype.

        A rank-k sketch of the sensitivities is k of these; the explicit
        Jacobian is nD of them, and is only affordable on small surveys --
        at 601,817 parameters it passes 32 GB at about 7,100 data.
        """
        width = self.chunk if chunk is None else chunk
        width = self.nS if not width else min(int(width), self.nS)
        solver = self._solver_for(width)
        if self._sr:
            self._ensure_primary()
        v_t = v.to(device=self.device, dtype=self.dtype).reshape(-1)
        with torch.no_grad():                      # one numeric factorization
            sig = torch.exp(-self._model_to_full(m)).unsqueeze(1)
            solver.factorize(torch.sparse.mm(self.Lstack, sig).view(self.nky, self.nnzA))
        out = torch.zeros_like(m)
        for s0, s1 in self._source_chunks(width):
            rows, gi, li = self._rows_of(s0, s1)
            mc = m.detach().clone().requires_grad_(True)
            m_full = self._model_to_full(mc)
            sig = torch.exp(-m_full).unsqueeze(1)
            VALS = torch.sparse.mm(self.Lstack, sig).view(self.nky, self.nnzA)
            U = self._solve_chunk(VALS, s0, s1, width, solver, prefactored=True,
                                  m_full=m_full)
            Ubar = (self.wq.view(-1, 1, 1) * U).sum(0)[:, :s1 - s0]
            d = torch.sparse.mm(self.Pd, Ubar)[gi, li]
            d.backward(v_t[rows])
            out = out + mc.grad
        return out

    def Jmatrix(self, m, block=32):
        """The explicit Jacobian (nD, nM), from one factorization.

        The adjoint right-hand side of the WHOLE Jacobian is known before the
        run: datum r reads row r of the receiver projection, so the nD adjoint
        sources are exactly the columns of Pd^T -- no model, no fields. The
        system is symmetric, so the factorization the forward already built
        serves them unchanged, and the nD adjoint solves collapse into
        ceil(nD/block) batched ones:

            dd_r/dA_k[e] = -w_k Y_k[row(e), r] U_k[col(e), s(r)]
            dd_r/dsigma  = sum_k L_k^T (that),   dd_r/dm = -sigma dd_r/dsigma

        with Y_k = A_k^-1 Pd^T. Against `Jtvec` called nD times this is the
        same matrix for two orders of magnitude less work, because a direct
        solve with many right-hand sides costs barely more than one: on a
        661-datum line, 84.8 s sequential against 0.31 s here (30.0 s for
        stock SimPEG's storeJ).

        `block` buys nothing above about 16 -- the solve saturates at that
        width while the fields it holds grow linearly with it (655 MiB at 32,
        5,413 MiB at 661). The default is the measured elbow.

        WHAT THIS DOES NOT BUY: J is dense and nD x nM however it is formed --
        30 MiB on that line, 5.2 GiB on a field survey, 31.8 GiB at 601,817
        parameters. This is the tool for the regime where the matrix fits
        (uncertainty quantification, Gauss-Newton curvature); at field scale
        `Jtvec` is still the only way through.

        m : (nM,) log-resistivity.  Returns (nD, nM) outside the autograd
        graph, on m's device and dtype.
        """
        if self._sr:
            raise NotImplementedError(
                "Jmatrix assumes the plain formulation. Under singularity "
                "removal the model also enters the right-hand side and the "
                "primary's background, terms this hand-written adjoint does "
                "not carry; use Jtvec, which differentiates the whole chain.")
        block = max(1, min(int(block), self.nD))
        dev0, dt0 = m.device, m.dtype
        with torch.no_grad():
            m_full = self._model_to_full(m)
            sig = torch.exp(-m_full)
            VALS = torch.sparse.mm(
                self.Lstack, sig.unsqueeze(1)).view(self.nky, self.nnzA)

            # the forward fields, whole survey: every block of data reads a
            # different set of source columns, so they are all needed at once
            fwd = self._solver_for(self.nS)
            fwd.factorize(VALS)
            U = fwd.solve(self.q)                      # (nky, nN, nS)
            row, col = fwd.row.long(), fwd.col.long()

            adj = self._solver_for(block)
            if adj is not fwd:
                adj.factorize(VALS)
            if self._PdT is None:
                self._PdT = self.Pd.t().coalesce().to_dense()   # (nN, nD)

            Jm = torch.empty(self.nC, self.nD, dtype=self.dtype,
                             device=self.device)
            for a in range(0, self.nD, block):
                b = min(a + block, self.nD)
                rhs = self._PdT[:, a:b]
                if b - a < block:        # the solver's plan is per rhs width
                    rhs = torch.cat(
                        [rhs, rhs.new_zeros(self.nN, block - (b - a))], 1)
                Y = adj.solve(rhs.contiguous())        # (nky, nN, block)
                cols = self.sidx[a:b]
                per_ky = [(-self.wq[k]) * Y[k][:, :b - a][row]
                          * U[k][:, cols][col] for k in range(self.nky)]
                Jm[:, a:b] = torch.sparse.mm(
                    self.Lstack.t(), torch.cat(per_ky, 0))
                del Y, per_ky
            Jm = -sig.unsqueeze(1) * Jm                # sigma = exp(-m)
            if self._P_inj is not None:
                if self._PinjT is None:
                    self._PinjT = self._P_inj.t().coalesce()
                Jm = torch.sparse.mm(self._PinjT, Jm)  # back to active cells
            return Jm.T.contiguous().to(device=dev0, dtype=dt0)

    def linearize(self, m, store_fields=None, fields_max_gib=2.0):
        """The forward frozen at m: one factorization, the fields kept, and
        J v / J^T v as one batched solve each. See `Linearization2D`."""
        return Linearization2D(self, m, store_fields=store_fields,
                               fields_max_gib=fields_max_gib)

    def Jvec(self, m, v):
        """J v for J = d(dpred)/dm, without forming J: one forward and one
        extra solve. For many products at one model, `linearize` once."""
        return self.linearize(m).Jvec(v)

    def misfit(self, m, dobs, w):
        """phi_d = ||w*(dpred-dobs)||^2 (no 1/2 factor — 0.25 convention)."""
        r = w * (self.dpred(m) - dobs)
        return (r ** 2).sum()


class Linearization2D:
    """The plain 2.5-D forward frozen at one model, for Gauss-Newton.

    A Gauss-Newton step solves (J^T D J + beta H_m) p = -g by conjugate
    gradients, which asks for J v and J^T v many times AT THE SAME MODEL. Done
    through `Jtvec` each product would refactorize and re-solve the forward;
    here the system is factorized once, the forward fields U_k are kept, and
    each product costs one batched solve over the nky systems:

        J v   :  A_k dU_k = -A_k(dsigma) U_k,     dsigma = -sigma v
                 J v = Pd sum_k w_k dU_k
        J^T v :  A_k Y_k = Pd^T v   (A is symmetric: the same factorization)
                 dd/dA_k[e] = -w_k Y_k[row e] . U_k[col e],  then L^T, -sigma

    which are the formulas `Jmatrix` already validates. The fields are kept on
    the device when they fit `fields_max_gib`; otherwise each product re-solves
    them from the stored factorization (two solves instead of one, still no
    refactorization).

    The solver object is shared with the engine, so anything else that
    factorizes it (a line-search `dpred`) replaces this factorization; the
    solver's `stamp` exposes that and the factorization is restored before
    the next product.
    """

    #: elements per gather chunk in the J^T products (three temporaries of
    #: this size live at once): 2M keeps them at ~16 MB each in float64
    gather_chunk = 2_000_000
    #: elements allowed in the per-row-block temporary of _rows_of_J
    #: (nky x nnz(A) x rows): the row block is sized to it, so a small mesh
    #: forms many rows per pass and a large one keeps few (16M elements gives
    #: 4 rows at 35k cells -- the earlier fixed value -- and ~40 at 6k)
    rows_budget = 16_000_000

    def __init__(self, eng, m, store_fields=None, fields_max_gib=2.0):
        if eng._sr:
            raise NotImplementedError(
                "Linearization2D assumes the plain formulation. Under "
                "singularity removal the model also enters the right-hand side "
                "and the primary's background, which these products do not "
                "carry; use Jtvec.")
        self.eng = eng
        self.m_dtype, self.m_device = m.dtype, m.device
        dev = eng.device
        w = eng.chunk if eng.chunk else eng.nS
        self.width = min(int(w), eng.nS)
        self.chunks = eng._source_chunks(self.width)
        self.solver = eng._solver_for(self.width)
        with torch.no_grad():
            self.sig = torch.exp(-eng._model_to_full(m))
            self.VALS = torch.sparse.mm(
                eng.Lstack, self.sig.unsqueeze(1)).view(eng.nky, eng.nnzA)
        self._factorize()

        self._crow = eng._crow.to(dev, torch.int64)
        self._col = eng._ccol.to(dev, torch.int64)
        self._row = self.solver.row.to(dev).long()
        # L^T is applied from L's own COO entries (see _Lt): a coalesced
        # transpose would be a second copy of the whole assembly map -- ~15M
        # entries, ~370 MB at 35k cells -- held for the life of every
        # linearization, plus the sort transient that builds it
        idx = eng.Lstack.indices()
        self._L_rows, self._L_cols = idx[0], idx[1]
        self._L_vals = eng.Lstack.values()
        if eng._PdTs is None:
            eng._PdTs = eng.Pd.t().coalesce()
        self._wq = eng.wq.view(-1, 1, 1)

        item = torch.empty((), dtype=eng.dtype).element_size()
        nbytes = eng.nky * eng.nN * self.width * len(self.chunks) * item
        if store_fields is None:
            store_fields = nbytes <= fields_max_gib * 2 ** 30
        self._U = [] if store_fields else None
        d = torch.empty(eng.nD, dtype=eng.dtype, device=dev)
        with torch.no_grad():
            for s0, s1 in self.chunks:
                U = self._solve_fields(s0, s1)
                if self._U is not None:
                    self._U.append(U)
                rows, gi, li = eng._rows_of(s0, s1)
                Ubar = (self._wq * U).sum(0)[:, :s1 - s0]
                d[rows] = torch.sparse.mm(eng.Pd, Ubar)[gi, li]
        self.d = d.to(self.m_device, self.m_dtype)

    # ---- the factorization and the fields ---------------------------------
    def _factorize(self):
        self.solver.factorize(self.VALS)
        self._stamp = self.solver.stamp

    def _ensure(self):
        if self.solver.stamp != self._stamp:
            self._factorize()

    def _solve(self, b):
        self._ensure()
        return self.solver.solve(b.contiguous())

    def _solve_fields(self, s0, s1):
        q = self.eng.q[:, s0:s1]
        if self.width > s1 - s0:
            q = torch.cat([q, q.new_zeros(self.eng.nN, self.width - (s1 - s0))], 1)
        return self._solve(q)

    def _fields(self, i):
        if self._U is not None:
            return self._U[i]
        return self._solve_fields(*self.chunks[i])

    def _to_eng(self, v):
        v = v.to(self.eng.device, self.eng.dtype).reshape(-1)
        if self.eng._P_inj is not None:
            v = torch.sparse.mm(self.eng._P_inj, v.unsqueeze(1)).squeeze(1)
        return v

    def _to_model(self, g_full):
        g = -self.sig * g_full                       # sigma = exp(-m)
        if self.eng._P_inj is not None:
            if self.eng._PinjT is None:
                self.eng._PinjT = self.eng._P_inj.t().coalesce()
            g = torch.sparse.mm(self.eng._PinjT, g.unsqueeze(1)).squeeze(1)
        return g.to(self.m_device, self.m_dtype)

    def _Lt(self, X, chunk=4_000_000):
        """L^T X for X (nky*nnzA,) or (nky*nnzA, k), straight from L's COO
        entries in chunks: (L^T X)[c] = sum over entries (r, c) of v * X[r].
        No transposed copy of L is ever built."""
        eng = self.eng
        shape = (eng.nC,) + tuple(X.shape[1:])
        out = torch.zeros(shape, dtype=X.dtype, device=X.device)
        n = self._L_vals.numel()
        step = max(chunk // (X.shape[1] if X.dim() > 1 else 1), 1)
        for p0 in range(0, n, step):
            p1 = min(p0 + step, n)
            v = self._L_vals[p0:p1]
            contrib = X[self._L_rows[p0:p1]]
            contrib *= v.unsqueeze(1) if X.dim() > 1 else v
            out.index_add_(0, self._L_cols[p0:p1], contrib)
        return out

    def _grad_vals(self, Y, U):
        """-sum_s Y_k[row e, s] U_k[col e, s] per stored entry, in chunks."""
        eng = self.eng
        out = torch.empty(eng.nky, eng.nnzA, dtype=eng.dtype, device=eng.device)
        step = max(self.gather_chunk // max(eng.nky * Y.shape[-1], 1), 1)
        for p0 in range(0, eng.nnzA, step):
            p1 = min(p0 + step, eng.nnzA)
            out[:, p0:p1] = -(Y[:, self._row[p0:p1]]
                              * U[:, self._col[p0:p1]]).sum(-1)
        return out

    # ---- the products -----------------------------------------------------
    @torch.no_grad()
    def Jvec(self, v):
        """J v, (nD,), on the model's device and dtype."""
        eng = self.eng
        dvals = torch.sparse.mm(
            eng.Lstack, (-self.sig * self._to_eng(v)).unsqueeze(1)
        ).view(eng.nky, eng.nnzA)
        out = torch.empty(eng.nD, dtype=eng.dtype, device=eng.device)
        for i, (s0, s1) in enumerate(self.chunks):
            U = self._fields(i)
            # -A_k(dsigma) U_k, written in place: no stacked copy, no negated copy
            rhs = torch.empty_like(U)
            for k in range(eng.nky):
                Ak = torch.sparse_csr_tensor(self._crow, self._col, -dvals[k],
                                             (eng.nN, eng.nN))
                rhs[k] = Ak @ U[k]
            dU = self._solve(rhs)
            del rhs
            rows, gi, li = eng._rows_of(s0, s1)
            dbar = (self._wq * dU).sum(0)[:, :s1 - s0]
            out[rows] = torch.sparse.mm(eng.Pd, dbar)[gi, li]
        return out.to(self.m_device, self.m_dtype)

    @torch.no_grad()
    def Jtvec(self, v):
        """J^T v, (nM,), on the model's device and dtype."""
        eng = self.eng
        v = v.to(eng.device, eng.dtype).reshape(-1)
        gsig = torch.zeros(eng.nC, dtype=eng.dtype, device=eng.device)
        for i, (s0, s1) in enumerate(self.chunks):
            rows, gi, li = eng._rows_of(s0, s1)
            V = torch.zeros(eng.nD, self.width, dtype=eng.dtype,
                            device=eng.device)
            V[gi, li] = v[rows]
            Y = self._solve(torch.sparse.mm(eng._PdTs, V))
            gv = self._grad_vals(Y, self._fields(i)) * eng.wq.view(-1, 1)
            gsig += self._Lt(gv.reshape(-1))
        return self._to_model(gsig)

    def _row_block(self, sub):
        """Rows of J formed per pass: `sub` if given, else sized to rows_budget."""
        if sub:
            return int(sub)
        per_row = max(self.eng.nky * self.eng.nnzA, 1)
        return int(max(1, min(self.width, self.rows_budget // per_row)))

    def _rows_of_J(self, sub):
        """Yield (a, b, Jb): rows a..b of J as an (nM, b - a) block, formed
        `sub` rows at a time from one adjoint solve per solver-width block of
        data. The temporary is nky x nnz(A) x `sub`, whatever the width."""
        eng = self.eng
        src0 = torch.tensor([s0 for s0, _ in self.chunks], device=eng.device)
        s0s = src0.tolist()
        sidx = eng.sidx
        for a in range(0, eng.nD, self.width):
            b = min(a + self.width, eng.nD)
            nb = b - a
            # columns a..b of Pd^T, from the sparse operator (no dense nN x nD)
            E = torch.zeros(eng.nD, self.width, dtype=eng.dtype, device=eng.device)
            E[torch.arange(a, b, device=eng.device),
              torch.arange(nb, device=eng.device)] = 1.0
            Y = self._solve(torch.sparse.mm(eng._PdTs, E))[:, :, :nb]
            del E
            # the forward field of each datum's own source, gathered per column
            src = sidx[a:b].tolist()
            ci = (torch.bucketize(sidx[a:b], src0, right=True) - 1).tolist()
            fields = {}
            cols = []
            for c, s_ in zip(ci, src):
                if c not in fields:
                    fields[c] = self._fields(c)
                cols.append(fields[c][:, :, s_ - s0s[c]])
            Us = torch.stack(cols, -1)
            del fields, cols
            sub_ = self._row_block(sub)
            for c0 in range(0, nb, sub_):
                c1 = min(c0 + sub_, nb)
                gv = self._grad_vals_pairwise(Y[..., c0:c1], Us[..., c0:c1])
                gv *= eng.wq.view(-1, 1, 1)
                Jb = self._Lt(gv.reshape(-1, c1 - c0))                    # (nC, sb)
                del gv
                Jb = -self.sig.unsqueeze(1) * Jb
                if eng._P_inj is not None:
                    if eng._PinjT is None:
                        eng._PinjT = eng._P_inj.t().coalesce()
                    Jb = torch.sparse.mm(eng._PinjT, Jb)
                yield a + c0, a + c1, Jb
            del Y, Us

    @torch.no_grad()
    def diag_JtDJ(self, dw, sub=None):
        """diag(J^T diag(dw) J), (nM,), exact, without storing J.

        Rows of J are formed `sub` at a time (`_rows_of_J`), squared, weighted,
        summed and dropped: the temporary is nky x nnz(A) x `sub` instead of x
        the solver width -- at 35k cells ~120 MB against ~740 MB, which is what
        decided the peak of matrix-free Gauss-Newton. Costs ceil(nD / width)
        solves.
        """
        dw = dw.to(self.eng.device, self.eng.dtype).reshape(-1)
        out = None
        for a, b, Jb in self._rows_of_J(sub):
            part = (Jb ** 2) @ dw[a:b]
            out = part if out is None else out + part
        return out.to(self.m_device, self.m_dtype)

    @torch.no_grad()
    def Jmatrix(self, sub=None):
        """The explicit J (nD, nM) from this linearization's factorization and
        fields, rows formed `sub` at a time: peak memory is J itself plus the
        linearization, not the nky x nnz(A) x block temporaries of
        `TorchDC2D.Jmatrix` (1.8 GiB against ~0.6 at 35k cells)."""
        J = None
        for a, b, Jb in self._rows_of_J(sub):
            if J is None:
                J = torch.empty(self.eng.nD, Jb.shape[0], dtype=self.eng.dtype,
                                device=self.eng.device)
            J[a:b] = Jb.T
        return J.to(self.m_device, self.m_dtype)

    def _grad_vals_pairwise(self, Y, U):
        """-Y_k[row e, r] U_k[col e, r] for each column r separately."""
        eng = self.eng
        nb = Y.shape[-1]
        out = torch.empty(eng.nky, eng.nnzA, nb, dtype=eng.dtype,
                          device=eng.device)
        step = max(self.gather_chunk // max(eng.nky * nb, 1), 1)
        for p0 in range(0, eng.nnzA, step):
            p1 = min(p0 + step, eng.nnzA)
            out[:, p0:p1] = -(Y[:, self._row[p0:p1]] * U[:, self._col[p0:p1]])
        return out
