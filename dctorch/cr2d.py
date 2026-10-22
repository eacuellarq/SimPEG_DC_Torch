"""
COMPLEX-RESISTIVITY 2.5D forward on GPU — the formulation the industry
standard actually uses (Kemna 2000; RES2DINV manual eq. 11.6; cR2/CRTomo),
implemented here as a THIRD forward alongside the linearized `TorchIP2D` and
the two-state nonlinear `TorchDCIP2D`
"""
import numpy as np
import torch

from .solver import make_batch_solver, sparse_solve_batch


class TorchCR2D:
    """Differentiable complex-resistivity 2.5D nodal forward.

    Parameters
    ----------
    mesh, survey : receivers must be plain "volt" (the ratio is formed here).
    ip_scale : multiplies d_ip (1000 for chargeability data in mV/V, which is
        also ~mrad of phase: rho* ~ (1/sigma_DC)(1 + i*m) => phase ~ m rad).
    dc_datum : "magnitude" (default, |V*|, the standard magnitude/phase pair)
        or "real" (Re V*).
    """

    def __init__(self, mesh, survey, nky=12, device="cuda",
                 dtype=torch.complex128, active_cells=None, val_inactive=None,
                 ip_scale=1.0, dc_datum="magnitude", quadrature=None):
        from .simulation2d import TorchDC2D

        rdt = torch.float64 if dtype == torch.complex128 else torch.float32
        dc = TorchDC2D(mesh, survey, nky=nky, device=device, dtype=rdt,
                       active_cells=active_cells, val_inactive=val_inactive,
                       quadrature=quadrature)
        self.dc = dc
        self.device, self.dtype, self.rdtype = dc.device, dtype, rdt
        self.nky, self.nD = dc.nky, dc.nD
        self.nP = dc.nC if dc._P_inj is None else dc._P_inj.shape[1]
        self.ip_scale = float(ip_scale)
        assert dc_datum in ("magnitude", "real"), dc_datum
        self.dc_datum = dc_datum

        # swap the real SPD batch solver for a COMPLEX SYMMETRIC one
        crow, col = dc.solver.crow, dc.solver.col
        dc.solver.free()
        dc.solver = None
        self.solver = make_batch_solver(crow, col, dc.nN, dc.nS, dc.nky,
                                        device=self.device, dtype=dtype,
                                        matrix_type="symmetric")
        self.q = dc.q.to(dtype)

    # ---------------- model <-> fields --------------------------------------
    def _inject_rho(self, m_rho):
        if self.dc._P_inj is None:
            return m_rho
        return torch.sparse.mm(self.dc._P_inj,
                               m_rho.unsqueeze(1)).squeeze(1) + self.dc._v_inj

    def _inject_ip(self, m_ip):
        if self.dc._P_inj is None:
            return m_ip
        return torch.sparse.mm(self.dc._P_inj, m_ip.unsqueeze(1)).squeeze(1)

    def split(self, m):
        m = m.reshape(-1)
        assert m.numel() == 2 * self.nP, \
            f"model must be 2*{self.nP} long, got {m.numel()}"
        return m[:self.nP], m[self.nP:]

    # ---------------- runtime (torch, differentiable) -----------------------
    def _spmm_real(self, M, v):
        """real sparse M @ real dense v (keeps everything out of complex mm)."""
        return torch.sparse.mm(M, v.unsqueeze(1)).squeeze(1)

    def dpred_from_fields(self, m_rho, m_ip):
        """(d_rho, d_ip) from log-rho_DC and chargeability m (model cells)."""
        dc = self.dc
        dev0, dt0 = m_rho.device, m_rho.dtype
        mr = m_rho.to(device=self.device, dtype=self.rdtype).reshape(-1)
        mi = m_ip.to(device=self.device, dtype=self.rdtype).reshape(-1)
        sig = torch.exp(-self._inject_rho(mr))          # sigma_DC (real)
        chg = self._inject_ip(mi)                       # m (real)
        # sigma* = sigma_DC (1 - i m):  two REAL spmm, then recombine
        vre = torch.sparse.mm(dc.Lstack, sig.unsqueeze(1)).view(dc.nky,
                                                                dc.nnzA)
        vim = torch.sparse.mm(dc.Lstack, (-sig * chg).unsqueeze(1)).view(
            dc.nky, dc.nnzA)
        VALS = torch.complex(vre, vim)
        U = sparse_solve_batch(VALS, self.q, self.solver)    # (nky, nN, nS)
        Ubar = (dc.wq.to(self.dtype).view(-1, 1, 1) * U).sum(0)
        # real projection applied to Re and Im separately
        Vr = torch.sparse.mm(dc.Pd, Ubar.real)[dc.iD, dc.sidx]
        Vi = torch.sparse.mm(dc.Pd, Ubar.imag)[dc.iD, dc.sidx]
        d_rho = torch.sqrt(Vr ** 2 + Vi ** 2) if self.dc_datum == "magnitude" \
            else Vr
        d_ip = self.ip_scale * Vi / Vr
        return d_rho.to(device=dev0, dtype=dt0), d_ip.to(device=dev0,
                                                         dtype=dt0)

    def dpred(self, m):
        """(d_rho, d_ip) for the stacked model m = cat(log rho_DC, m)."""
        m_rho, m_ip = self.split(m)
        return self.dpred_from_fields(m_rho, m_ip)

    def misfit(self, m, dobs_rho, w_rho, dobs_ip, w_ip, lam=1.0):
        """phi_d^rho + lam * phi_d^IP (no 1/2 — the 0.25 convention)."""
        d_r, d_i = self.dpred(m)
        return ((w_rho * (d_r - dobs_rho)) ** 2).sum() \
            + lam * ((w_ip * (d_i - dobs_ip)) ** 2).sum()
