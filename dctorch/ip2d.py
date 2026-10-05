"""
Linearized 2.5D induced-polarization forward on GPU (Seigel), on top of
`TorchDC2D`, matching STOCK SimPEG's `induced_polarization.Simulation2DNodal`
convention (extracted from simpeg 0.25.2 source, not assumed):

    d_IP = scale * J_sigma[sig_bg] @ (dsigma/deta) @ eta,
    dsigma/deta = -diag(sig_bg)              (BaseIPSimulation.sigmaDeriv)
    scale       = 1 / V_dc  per datum whose receiver has
                  data_type == "apparent_chargeability", else 1
                                             (BaseIPSimulation._scale)

which per wavenumber ky is one EXTRA solve reusing the background
factorization (same machinery as the adjoint):

    vals(dA_ky) = L_ky @ dsigma          (L_ky maps ALREADY extracted by DC)
    du_ky       = -A_ky^{-1} (dA_ky @ u_ky)
    d_IP        = scale * P sum_j w_j du_ky_j

Everything is linear in eta, so autograd traverses it trivially and the
eta-inversion is a linear problem under the standard protocol. The wrapped
`TorchDC2D` is factorized ONCE on the background conductivity in __init__;
dpred(eta) does solves only (no refactorization). Do not share the wrapped
engine: calling its dpred would refactorize away the background.

Model convention: eta in the same units as the observed data implies (the
forward is linear, so mV/V data yield eta in mV/V — UBC practice); the
background model m_dc = log(rho) full-mesh, matching TorchDC2D.
"""
import numpy as np
import torch


class _FixedFactorSolveFn(torch.autograd.Function):
    """x_i = A_i^{-1} b_i with A_i FIXED (already factorized, SPD):
    differentiable in b only; backward is one adjoint solve on the same
    factorization."""

    @staticmethod
    def forward(ctx, b, solver):
        x = solver.solve(b.detach())
        ctx.solver = solver
        ctx.dt_b = b.dtype
        return x

    @staticmethod
    def backward(ctx, grad_x):
        adj = ctx.solver.solve(grad_x.contiguous())
        return adj.to(ctx.dt_b), None


class TorchIP2D:
    """Differentiable linearized-IP dpred/misfit for 2.5D nodal on a fixed
    mesh+survey+background.

    Parameters
    ----------
    mesh, survey : the IP survey (receivers may carry
        data_type="apparent_chargeability"; the per-datum 1/V_dc scale is
        applied exactly like stock's BaseIPSimulation._scale).
    m_dc : (nC,) numpy log-rho background on the FULL mesh (e.g. the
        recovered DC model); sigma_bg = exp(-m_dc).
    active_cells : optional bool mask for the ETA model (inactive eta = 0).
    """

    def __init__(self, mesh, survey, m_dc, nky=12, device="cuda",
                 dtype=torch.float64, active_cells=None, quadrature=None):
        from .simulation2d import TorchDC2D

        dc = TorchDC2D(mesh, survey, nky=nky, device=device, dtype=dtype,
                       quadrature=quadrature)
        self.dc = dc
        self.device, self.dtype = dc.device, dc.dtype

        self._P_inj = None
        if active_cells is not None:
            act = np.asarray(active_cells, dtype=bool)
            ia = np.where(act)[0]
            self._P_inj = torch.sparse_coo_tensor(
                torch.tensor(np.vstack((ia, np.arange(ia.size))),
                             dtype=torch.int64, device=self.device),
                torch.ones(ia.size, dtype=self.dtype, device=self.device),
                (mesh.nC, ia.size)).coalesce()

        # ---- background: factorize ONCE, keep fields and 1/V_dc scale ----
        m_bg = np.asarray(m_dc, dtype=float).reshape(-1)
        assert m_bg.size == mesh.nC, "m_dc must live on the full mesh"
        sig = torch.tensor(np.exp(-m_bg), dtype=self.dtype,
                           device=self.device)
        self._sig_bg = sig
        vals = torch.sparse.mm(dc.Lstack, sig.unsqueeze(1)).view(
            dc.nky, dc.nnzA)
        dc.solver.factorize(vals)
        with torch.no_grad():
            U = dc.solver.solve(dc.q)                    # (nky, nN, nS)
        self._U_bg = U
        Ubar = (dc.wq.view(-1, 1, 1) * U).sum(0)
        v_dc = torch.sparse.mm(dc.Pd, Ubar)[dc.iD, dc.sidx]
        self.v_dc = v_dc

        # per-datum scale, mirroring BaseIPSimulation._scale: 1/V_dc on
        # apparent_chargeability receiver blocks (survey order == Pd order)
        scale = torch.ones(dc.nD, dtype=self.dtype, device=self.device)
        i0 = 0
        for src in survey.source_list:
            for rx in src.receiver_list:
                i1 = i0 + rx.nD
                if rx.data_type == "apparent_chargeability":
                    scale[i0:i1] = 1.0 / v_dc[i0:i1]
                i0 = i1
        assert i0 == dc.nD
        self._scale = scale
        # int64 copies for tensor indexing in the per-ky gather
        self._row = dc.solver.row.to(torch.int64)
        self._col = dc.solver.col.to(torch.int64)

    # ---------------- runtime (torch, differentiable, linear in eta) -------
    def dpred(self, eta):
        """Predicted IP data (nD,) for eta on the full mesh (or active cells
        when built with active_cells). Autograd-traceable; linear."""
        dc = self.dc
        dev0, dt0 = eta.device, eta.dtype
        e = eta.to(device=self.device, dtype=self.dtype).reshape(-1)
        if self._P_inj is not None:
            e = torch.sparse.mm(self._P_inj, e.unsqueeze(1)).squeeze(1)
        dsig = -self._sig_bg * e                     # stock sigmaDeriv
        DVALS = torch.sparse.mm(dc.Lstack, dsig.unsqueeze(1)).view(
            dc.nky, dc.nnzA)
        row, col = self._row, self._col
        rhs = []
        for i in range(dc.nky):                      # dA_ky @ u_ky (gather)
            contrib = DVALS[i].unsqueeze(1) * self._U_bg[i][col]
            r = torch.zeros(dc.nN, dc.nS, dtype=self.dtype,
                            device=self.device).index_add_(0, row, contrib)
            rhs.append(r)
        dU = _FixedFactorSolveFn.apply(-torch.stack(rhs), dc.solver)
        dUbar = (dc.wq.view(-1, 1, 1) * dU).sum(0)
        d = torch.sparse.mm(dc.Pd, dUbar)[dc.iD, dc.sidx] * self._scale
        return d.to(device=dev0, dtype=dt0)

    def misfit(self, eta, dobs, w):
        """phi_d = ||w*(dpred-dobs)||^2 (no 1/2 factor — 0.25 convention)."""
        r = w * (self.dpred(eta) - dobs)
        return (r ** 2).sum()
