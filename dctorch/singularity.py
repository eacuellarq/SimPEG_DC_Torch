"""
Singularity removal for the 2.5D nodal DC problem.

The potential of a point source diverges at the electrode, and a mesh resolves
that divergence badly: the error of the total-field solve is concentrated on the
few cells around the injection, which is why DC meshes are refined at the
electrodes rather than where the model varies. Splitting the field into a part
known in closed form and a smooth remainder moves the divergence out of the
discretization:

    u = u_p + u_s,      A(sigma) u_s = -(A(sigma) - A(sigma_p)) u_p

with u_p the analytic potential of a homogeneous half space of resistivity
rho_p. In the wavenumber domain that potential is

    u_p(ky, r) = (rho_p / pi) K_0(ky r)

(the 1/(2 pi) of the whole space doubled by the free-surface image), and at the
source node, where r = 0, the mean of K_0 over a disc of the same area as the
cell replaces the divergence:

    <K_0>_disc(a) = 2 [1 - a K_1(a)] / a^2,     a = ky R,  R = sqrt(hx hz / pi)

which is the exact integral, not a fitted constant.

Two properties make this cheap inside an inversion: u_p does not depend on the
model, so it is built once and reused at every iteration, and the right-hand
side above is linear in the system values, so the whole path stays autograd
traceable and the adjoint is unchanged.

The background resistivity is not one number for the survey. Because the
assembly is linear in sigma, a half space of resistivity rho_s has potential
rho_s * u_1 with u_1 the same geometric factor for every source, and

    A(1/rho_s) (rho_s u_1) = A(1) u_1

for any rho_s at all. So the geometric part is built once, each source scales it
by the resistivity the current model carries at its own electrode, and the fixed
side of the right-hand side is cached once as well. A background that follows the
model matters wherever the near surface is not uniform: there a single global
rho_p leaves a large contrast exactly at the electrode, which is where the
primary is singular and where the split is supposed to help.

The primary assumes a flat free surface. With topography the closed form no
longer satisfies the boundary condition and the split loses its accuracy.
"""
import numpy as np
import torch


def primary_potential(mesh, survey, kys, rho_p, dtype=np.float64):
    """Analytic half-space potential at every node: (nky, n_nodes, n_sources).

    rho_p may be a scalar or one resistivity per source.
    """
    from scipy.special import k0, k1

    nodes = np.asarray(mesh.nodes, dtype=float)
    kys = np.asarray(kys, dtype=float)
    src = survey.source_list
    rho = np.broadcast_to(np.atleast_1d(np.asarray(rho_p, float)), (len(src),))

    # radius of the equal-area disc of the cell holding each source
    h = np.asarray(mesh.h_gridded, dtype=float)

    out = np.zeros((kys.size, nodes.shape[0], len(src)), dtype=dtype)
    for j, s in enumerate(src):
        locs = [(np.asarray(s.location_a, float), 1.0)]
        if getattr(s, "location_b", None) is not None:
            locs.append((np.asarray(s.location_b, float), -1.0))
        for loc, sign in locs:
            r = np.linalg.norm(nodes - loc[None, :2], axis=1)
            at_source = r < 1e-9 * max(1.0, float(np.abs(nodes).max()))
            ic = int(np.atleast_1d(mesh.point2index(loc[None, :2]))[0])  # TreeMesh
            R = float(np.sqrt(np.prod(h[ic]) / np.pi))                   # returns a scalar
            for ik, ky in enumerate(kys):
                v = k0(np.maximum(ky * r, 1e-300)) / np.pi
                if at_source.any():
                    a = ky * R
                    v[at_source] = 2.0 * (1.0 - a * k1(a)) / a ** 2 / np.pi
                out[ik, :, j] += sign * rho[j] * v
    return out


class _SpmmValues(torch.autograd.Function):
    """A(v) @ u where only the values v of a fixed pattern carry a gradient.

    torch's own sparse matmul differentiates its values through a dense outer
    product of the two factors, which for this operator is a matrix the size of
    the mesh squared. The derivative is one number per stored entry,

        d/dv_e  <g, A(v) u>  =  g[row_e] . u[col_e],

    so the backward costs one gather per entry and the forward keeps the sparse
    product untouched.
    """

    @staticmethod
    def forward(ctx, vals, idx, u, shape):
        ctx.save_for_backward(vals, idx, u)
        ctx.shape = shape
        A = torch.sparse_coo_tensor(idx, vals, shape, is_coalesced=True)
        return torch.sparse.mm(A, u)

    @staticmethod
    def backward(ctx, g):
        vals, idx, u = ctx.saved_tensors
        g = g.contiguous()
        gv = (g[idx[0]] * u[idx[1]]).sum(1) if ctx.needs_input_grad[0] else None
        gu = None
        if ctx.needs_input_grad[2]:      # A is symmetric here, so A^T g is A g
            A = torch.sparse_coo_tensor(idx, vals, ctx.shape, is_coalesced=True)
            gu = torch.sparse.mm(A, g)
        return gv, None, gu, None


class Primary:
    """The model-independent half of a singularity-removed solve.

    Holds the analytic potential on the host (it is as large as the fields of the
    whole survey) and hands out one source chunk at a time, together with the
    system values of the background it belongs to.
    """

    def __init__(self, eng, mesh, survey, rho_p=None):
        self.fixed = None if rho_p is None else float(rho_p)
        up = primary_potential(mesh, survey, eng.kys, 1.0)      # geometry only
        self.up = torch.tensor(up, dtype=eng.dtype).pin_memory()

        # the cell each source sits in, so the model can supply its own background
        self.src_cell = torch.tensor(
            [int(np.atleast_1d(mesh.point2index(
                np.asarray(s.location_a, float)[None, :2]))[0])
             for s in survey.source_list], dtype=torch.int64, device=eng.device)

        ones = torch.ones((mesh.n_cells, 1), dtype=eng.dtype, device=eng.device)
        self.vals_one = torch.sparse.mm(eng.Lstack, ones).view(eng.nky, eng.nnzA)

        # one block-diagonal pattern over the wavenumbers: the right-hand sides of
        # all nky systems are then a single sparse product, not nky of them
        crow = eng._crow.to(torch.int64)
        rows = torch.repeat_interleave(torch.arange(eng.nN, dtype=torch.int64),
                                       crow[1:] - crow[:-1])
        cols = eng._ccol.to(torch.int64)
        off = (torch.arange(eng.nky, dtype=torch.int64) * eng.nN).repeat_interleave(rows.numel())
        self.idx = torch.stack([rows.repeat(eng.nky) + off,
                                cols.repeat(eng.nky) + off]).to(eng.device)
        self.shape = (eng.nky * eng.nN, eng.nky * eng.nN)
        self.nN, self.nky, self.device = eng.nN, eng.nky, eng.device

    def chunk(self, s0, s1, width=None):
        """Unit primary of sources [s0, s1), padded to `width` columns."""
        u = self.up[:, :, s0:s1].to(self.device, non_blocking=True)
        if width and width > s1 - s0:
            u = torch.cat([u, u.new_zeros(u.shape[0], u.shape[1], width - (s1 - s0))], dim=2)
        return u

    def scale(self, m_full, s0, s1, width=None):
        """Background resistivity of each source: from the model at its own
        electrode, unless a fixed one was asked for."""
        if self.fixed is not None:
            r = torch.full((s1 - s0,), self.fixed, dtype=self.vals_one.dtype,
                           device=self.device)
        else:
            r = torch.exp(m_full[self.src_cell[s0:s1]])
        if width and width > s1 - s0:
            r = torch.cat([r, r.new_ones(width - (s1 - s0))])
        return r.view(1, 1, -1)

    def rhs(self, vals, u, rho_s):
        """A(1) u_1 - A(sigma) (rho_s u_1), which is -(A(sigma) - A(sigma_p)) u_p
        exactly, for a background that may differ from source to source.

        Each wavenumber borrows the system's own sparsity pattern, already sorted
        by row, so the tensor can be declared coalesced and the product never
        materializes the nnz-by-sources array a scatter would need -- which is
        what the peak memory of this path is decided by.
        """
        flat = u.reshape(self.nky * self.nN, -1)
        ref = _SpmmValues.apply(self.vals_one.reshape(-1), self.idx, flat, self.shape)
        cur = _SpmmValues.apply(vals.reshape(-1), self.idx,
                                (u * rho_s).reshape(self.nky * self.nN, -1), self.shape)
        return (ref - cur).view_as(u)
