"""
Linearized 3D induced-polarization forward on GPU (Seigel), on top of
`TorchDC3D`, matching STOCK SimPEG's `induced_polarization.Simulation3DNodal`
convention (same as the 2.5D case, extracted from 0.25.2 source):

    d_IP = scale * J_sigma[sig_bg] @ (-diag(sig_bg)) @ eta
    scale = 1 / V_dc  per apparent_chargeability datum, else 1

i.e. ONE extra multi-RHS solve reusing the background factorization:

    vals(dA) = L3 @ dsigma,  du = -A^{-1} (dA @ u_bg),  d_IP = scale * P du

Differences vs the 2.5D wrapper are scale-driven: no wavenumber batch, and
dA @ u_bg goes through a custom autograd Function (`_SpmvFixedU`) whose
forward/backward are CHUNKED gathers on the sparsity pattern — the naive
index_add graph would retain nnzA x nS intermediates (~10 GB on the field
mesh). The wrapped `TorchDC3D` is factorized ONCE on the background in
__init__; dpred(eta) does solves only. Do not share the wrapped engine.

Model convention: eta on the active cells (inactive eta = 0) in the units of
the data (mV/V data => eta in mV/V); background m_dc = log(rho) on the active
cells (injected with the DC engine's own val_inactive), or full mesh when no
active_cells are given.
"""
import os

import numpy as np
import torch

from .ip2d import _FixedFactorSolveFn

# elements per gather chunk in the dA@U product (~64 MB of f64 intermediates)
_SPMV_CHUNK_ELEMS = int(os.environ.get("DCTORCH_SPMV_CHUNK", 8_000_000))


class _SpmvFixedU(torch.autograd.Function):
    """rhs = dA(vals) @ U with U FIXED: rhs[r] = sum_p vals[p] U[c_p] over the
    pattern (row, col). Forward and backward are chunked gathers; nothing
    (nnz x nS)-sized is ever retained."""

    @staticmethod
    def forward(ctx, vals, U, row, col):
        n, k = U.shape
        rhs = torch.zeros(n, k, dtype=U.dtype, device=U.device)
        step = max(_SPMV_CHUNK_ELEMS // max(k, 1), 1)
        with torch.no_grad():
            for p0 in range(0, vals.numel(), step):
                p1 = min(p0 + step, vals.numel())
                rhs.index_add_(0, row[p0:p1],
                               vals[p0:p1].unsqueeze(1) * U[col[p0:p1]])
        ctx.save_for_backward(U, row, col)
        return rhs

    @staticmethod
    def backward(ctx, grad_rhs):
        U, row, col = ctx.saved_tensors
        k = U.shape[1]
        nnz = row.numel()
        grad_vals = torch.empty(nnz, dtype=U.dtype, device=U.device)
        step = max(_SPMV_CHUNK_ELEMS // max(k, 1), 1)
        for p0 in range(0, nnz, step):
            p1 = min(p0 + step, nnz)
            grad_vals[p0:p1] = (grad_rhs[row[p0:p1]]
                                * U[col[p0:p1]]).sum(-1)
        return grad_vals, None, None, None


class TorchIP3D:
    """Differentiable linearized-IP dpred/misfit for 3D nodal on a fixed
    mesh+survey+background.

    Parameters
    ----------
    mesh, survey : the IP survey (receivers may carry
        data_type="apparent_chargeability" -> per-datum 1/V_dc scale, exactly
        stock's BaseIPSimulation._scale).
    m_dc : numpy log-rho background — on the ACTIVE cells when active_cells
        is given (injected with val_inactive, like the DC inversions), else
        on the full mesh.
    active_cells : optional bool mask; eta lives on the active cells
        (inactive eta = 0).
    val_inactive : log-rho of inactive cells for the BACKGROUND (e.g.
        np.log(1e8) air), required with active_cells.
    """

    def __init__(self, mesh, survey, m_dc, device="cuda",
                 dtype=torch.float64, active_cells=None, val_inactive=None,
                 bc_type="Robin"):
        from .simulation3d import TorchDC3D

        dc = TorchDC3D(mesh, survey, device=device, dtype=dtype,
                       active_cells=active_cells, val_inactive=val_inactive,
                       bc_type=bc_type)
        self.dc = dc
        self.device, self.dtype = dc.device, dc.dtype

        # ---- background: factorize ONCE, keep fields and 1/V_dc scale ----
        m_bg = torch.tensor(np.asarray(m_dc, dtype=float).reshape(-1),
                            dtype=self.dtype, device=self.device)
        if dc._P_inj is not None:
            assert m_bg.numel() == dc._P_inj.shape[1], \
                "m_dc must live on the active cells"
            m_full = torch.sparse.mm(
                dc._P_inj, m_bg.unsqueeze(1)).squeeze(1) + dc._v_inj
        else:
            assert m_bg.numel() == mesh.nC, "m_dc must live on the full mesh"
            m_full = m_bg
        sig = torch.exp(-m_full)
        self._sig_bg = sig
        vals = torch.sparse.mm(dc.L3, sig.unsqueeze(1)).squeeze(1)
        dc.solver.factorize(vals)
        with torch.no_grad():
            U = dc.solver.solve(dc.q)                    # (nN, nS)
        self._U_bg = U
        v_dc = torch.sparse.mm(dc.Pd, U)[dc.iD, dc.sidx]
        self.v_dc = v_dc

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
        # int64 pattern for the chunked gathers
        self._row = dc.solver.row.to(torch.int64)
        self._col = dc.solver.col.to(torch.int64)

    # ---------------- runtime (torch, differentiable, linear in eta) -------
    def dpred(self, eta):
        """Predicted IP data (nD,) for eta on the active cells (or full mesh
        when built without active_cells). Autograd-traceable; linear."""
        dc = self.dc
        dev0, dt0 = eta.device, eta.dtype
        e = eta.to(device=self.device, dtype=self.dtype).reshape(-1)
        if dc._P_inj is not None:
            e = torch.sparse.mm(dc._P_inj, e.unsqueeze(1)).squeeze(1)
        dsig = -self._sig_bg * e                     # stock sigmaDeriv
        dvals = torch.sparse.mm(dc.L3, dsig.unsqueeze(1)).squeeze(1)
        rhs = _SpmvFixedU.apply(dvals, self._U_bg, self._row, self._col)
        dU = _FixedFactorSolveFn.apply(-rhs, dc.solver)
        d = torch.sparse.mm(dc.Pd, dU)[dc.iD, dc.sidx] * self._scale
        return d.to(device=dev0, dtype=dt0)

    def misfit(self, eta, dobs, w):
        """phi_d = ||w*(dpred-dobs)||^2 (no 1/2 factor — 0.25 convention)."""
        r = w * (self.dpred(eta) - dobs)
        return (r ** 2).sum()
