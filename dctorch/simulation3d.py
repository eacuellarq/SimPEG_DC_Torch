"""
Differentiable 3D DC-resistivity forward, driven by STOCK SimPEG 0.25.

Same recipe as the 2.5D engine minus the wavenumber batch: at build time the
stock `Simulation3DNodal` assembly is collapsed into ONE sparse linear map

    A(sigma).data = L3 @ sigma,   L3 = T1 @ Qe + Td @ AvgBC

(both the FV term and the Robin-BC diagonal are linear in sigma), validated
against `sim.getA()` at two models before use. A forward is then: one sparse
matmul + one multi-RHS direct solve (cuDSS on cuda / SuperLU on cpu) + one
stacked receiver projection — fully autograd-traceable (exact vjp Jtvec).

Ported from the fork's `Simulation3DNodal._get_L3` + `_dpred_torch_fast`
(validated 1e-16 / 117 s on a field survey); here it lives OUTSIDE simpeg.
Model convention: m = log(rho); active_cells fixes the air like the 2.5D.
"""
import numpy as np
import scipy.sparse as sp
import torch

from .solver import make_solver, sparse_solve


class _SpMV(torch.autograd.Function):
    """y = M x for a fixed sparse M whose transpose is cached alongside it.

    Mathematically identical to `torch.sparse.mm(M, x)` -- verified bit-exact --
    but the vjp multiplies by a transpose that was built once instead of one
    that torch reconstructs on every backward pass.
    """

    @staticmethod
    def forward(ctx, x, M, Mt):
        ctx.Mt = Mt
        return torch.sparse.mm(M, x.unsqueeze(1)).squeeze(1)

    @staticmethod
    def backward(ctx, g):
        gx = torch.sparse.mm(ctx.Mt, g.contiguous().unsqueeze(1)).squeeze(1)
        return gx, None, None


class TorchDC3D:
    """Differentiable dpred/misfit for 3D nodal DC on a fixed mesh+survey."""

    def __init__(self, mesh, survey, device="cuda", dtype=torch.float64,
                 active_cells=None, val_inactive=None, bc_type="Robin"):
        from simpeg import maps
        from simpeg.electromagnetics.static import resistivity as dc

        self.device = torch.device(device)
        self.dtype = dtype
        ext = dc.simulation.Simulation3DNodal(
            mesh, survey=survey, sigmaMap=maps.IdentityMap(nP=mesh.nC),
            bc_type=bc_type)
        self._extract(ext, mesh, survey)
        self._P_inj = self._P_injt = None
        if active_cells is not None:
            act = np.asarray(active_cells, dtype=bool)
            ia = np.where(act)[0]
            P = sp.csr_matrix(
                (np.ones(ia.size), (ia, np.arange(ia.size))),
                shape=(mesh.nC, ia.size))
            self._P_inj = self._s2t(P, self.device, self.dtype)
            self._P_injt = self._s2t(sp.csr_matrix(P.T), self.device, self.dtype)
            v = np.full(mesh.nC, float(val_inactive))
            v[act] = 0.0
            self._v_inj = torch.tensor(v, dtype=self.dtype,
                                       device=self.device)

    @staticmethod
    def _s2t(M, dev=None, dt=None):
        """Module-level twin of the local `s2t`, for operators built outside
        `_extract` (the active-cell injection)."""
        M = M.tocsr()
        M.sort_indices()
        return torch.sparse_csr_tensor(
            torch.tensor(M.indptr, dtype=torch.int32, device=dev),
            torch.tensor(M.indices, dtype=torch.int32, device=dev),
            torch.tensor(M.data, dtype=dt, device=dev),
            size=M.shape, device=dev)

    def _extract(self, ext, mesh, survey):
        nC, nN, nE = mesh.nC, mesh.n_nodes, mesh.n_edges
        sig0 = np.ones(nC)
        ext.model = sig0
        A0 = sp.csr_matrix(ext.getA())      # also populates _AvgBC via setBC
        A0.sort_indices()
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

        Grad = sp.csr_matrix(mesh.nodal_gradient)
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
              np.concatenate(eI_l))), shape=(nnzA, nE)).tocsr()

        Me_test = sp.csr_matrix(mesh.get_edge_inner_product(model=sig0))
        assert (Me_test - sp.diags(Me_test.diagonal())).nnz == 0, \
            "MeSigma not diagonal"
        Qe = sp.csr_matrix(
            mesh.get_edge_inner_product_deriv(np.ones(nC))(np.ones(nE)))
        Td = sp.coo_matrix(
            (np.ones(nN), (pos_of(np.arange(nN), np.arange(nN)),
                           np.arange(nN))), shape=(nnzA, nN)).tocsr()
        L3 = sp.csr_matrix(T1 @ Qe + Td @ sp.csr_matrix(ext._AvgBC))

        # validate at two models before trusting the map
        rng = np.random.RandomState(7)
        for sig in (sig0 * 1e-2, 1e-2 * np.exp(0.5 * rng.randn(nC))):
            ext.model = sig
            Aref = sp.csr_matrix(ext.getA())
            Aref.sort_indices()
            err = np.abs(L3 @ sig - Aref.data).max() / np.abs(Aref.data).max()
            assert err < 1e-10, f"L3 validation failed: {err:.2e}"

        q = np.asarray(ext.getRHS().todense()
                       if sp.issparse(ext.getRHS()) else ext.getRHS())
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

        dev, dt = self.device, self.dtype

        def s2t(M):
            """scipy CSR -> torch CSR, int32 indices, no intermediate COO.

            The previous route went through `sparse_coo_tensor(...).coalesce()`,
            which sorts on the device and therefore holds two full copies of the
            operator at once; the discarded one stays in torch's caching
            allocator, where cuDSS -- which allocates with raw cudaMalloc --
            cannot reach it. On the dh=8 mesh L3 carries 51.7M non-zeros, so
            that transient is about 1 GiB on a card that has 8. Building CSR
            straight from the scipy CSR skips both the copy and the sort, and
            int32 indices (what scipy already uses) halve the index storage
            against the int64 torch defaults.
            """
            M = M.tocsr()
            M.sort_indices()
            return torch.sparse_csr_tensor(
                torch.tensor(M.indptr, dtype=torch.int32, device=dev),
                torch.tensor(M.indices, dtype=torch.int32, device=dev),
                torch.tensor(M.data, dtype=dt, device=dev),
                size=M.shape, device=dev)

        self.nC, self.nN, self.nS, self.nD = nC, nN, nS, nD
        self.nnzA = nnzA
        self.L3 = s2t(L3)
        # The vjp of `L3 @ sigma` needs L3^T. torch rebuilds a sparse transpose
        # on every backward (transpose + sort of 9.9M non-zeros at dh=20:
        # measured 38.5 ms against 1.8 for the forward, and ~200 MiB of scratch
        # each time). Transposing once on the host, where it is free, removes
        # both -- see `_SpMV` below.
        self.L3t = s2t(sp.csr_matrix(L3).T)
        self.q = torch.tensor(q, dtype=dt, device=dev)
        self.Pd = s2t(Pd)
        self.sidx = torch.tensor(np.asarray(sidx), dtype=torch.int64,
                                 device=dev)
        self.iD = torch.arange(nD, dtype=torch.int64, device=dev)
        self.solver = make_solver(
            torch.tensor(A0.indptr, dtype=torch.int32),
            torch.tensor(A0.indices, dtype=torch.int32),
            nN, nS, device=dev, dtype=dt)

    def dpred(self, m):
        """Predicted data (nD,) for m = log(rho). Autograd-traceable."""
        dev0, dt0 = m.device, m.dtype
        m_loc = m.to(device=self.device, dtype=self.dtype).reshape(-1)
        if self._P_inj is not None:
            m_loc = _SpMV.apply(m_loc, self._P_inj, self._P_injt) + self._v_inj
        sig = torch.exp(-m_loc)
        VALS = _SpMV.apply(sig, self.L3, self.L3t)
        U = sparse_solve(VALS, self.q, self.solver)          # (nN, nS)
        d = torch.sparse.mm(self.Pd, U)[self.iD, self.sidx]
        return d.to(device=dev0, dtype=dt0)

    def Jtvec(self, m, v):
        """J^T v for J = d(dpred)/dm, without forming J.

        The reverse pass of `dpred` with an arbitrary data-space vector in
        place of the residual: one extra simulation, no nD x nM allocation.
        Unlike the 2.5-D engine this one has no source chunking yet, so the
        whole survey's fields are held at once.

        m : (nM,) log-resistivity.  v : (nD,) any data-space vector.
        """
        mc = m.detach().clone().requires_grad_(True)
        d = self.dpred(mc)
        v_t = v.to(device=d.device, dtype=d.dtype).reshape(-1)
        return torch.autograd.grad(d, mc, grad_outputs=v_t)[0]

    def misfit(self, m, dobs, w):
        """phi_d = ||w*(dpred-dobs)||^2 (0.25 convention)."""
        r = w * (self.dpred(m) - dobs)
        return (r ** 2).sum()
