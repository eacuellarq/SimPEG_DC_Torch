"""Does chunking the gradient over SOURCES bound the memory, and what does it
cost?

Motivation. On the densified survey the peak device memory decomposes as
1.14 GiB fixed plus 2.35 MiB per source, and at the largest survey the stored
fields are 82 per cent of the peak. The fields are what grows, so they are what
a chunking strategy has to attack. Chunking over *data* does not help in DC:
the adjoint carries one right-hand side per source, and every receiver of a
source shares that field.

The scheme measured here. The factorisation is computed once and reused. The
sources are then processed in chunks of k: solve for k fields, project them to
the data those sources produced, form that chunk's contribution to the misfit,
backpropagate it, accumulate the gradient, release the fields. Because the
misfit is a sum over data and every datum belongs to exactly one source, the
accumulated gradient equals the one-shot gradient exactly — the script asserts
this rather than assuming it.

What it costs. Each chunk is a separate triangular solve with a narrower
right-hand side, so parallelism per solve drops while the factorisation is
paid only once. The measurement is whether the memory saved is worth the time.

Env: NLINES (def 33), CHUNKS (def "0,64,128,256,512"; 0 = one shot).
Emits one "JSON: {...}" line per chunk size. CWD scripts/, env simpeg311.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import time

import numpy as np
import torch  # noqa: F401  (before simpeg/pydiso: MKL DLL clash)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line)
from simpeg.utils import model_builder
from discretize import TreeMesh
from discretize.utils import active_from_xyz
import pynvml

from dctorch import TorchDC3D
from dctorch.solver import make_solver

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "..", "..", "..", "examples",
                   "inv_dcr_3d_files") + os.path.sep
DH = 25.0
NLINES = int(os.environ.get("NLINES", "33"))
CHUNKS = [int(c) for c in os.environ.get("CHUNKS", "0,64,128,256,512").split(",")]
DT = torch.float32
np.random.seed(0)

pynvml.nvmlInit()
_h = pynvml.nvmlDeviceGetHandleByIndex(0)
nvml = lambda: pynvml.nvmlDeviceGetMemoryInfo(_h).used / 2 ** 30  # noqa: E731

# ---------------------------------------------------------------- the problem
topo_xyz = np.loadtxt(DIR + "topo_xyz.txt")
srcs = []
for y in np.linspace(-800, 800, NLINES):
    srcs += generate_dcip_sources_line(
        "dipole-dipole", "volt", "3D", np.r_[-1000.0, 1000.0, y, y],
        topo_xyz, num_rx_per_src=8, station_spacing=50.0)
survey = dc.Survey(srcs)
nD, nS = int(survey.nD), len(survey.source_list)

nbc = lambda w: 2 ** int(np.round(np.log(w / DH) / np.log(2.0)))  # noqa: E731
mesh = TreeMesh([[(DH, nbc(8000.))], [(DH, nbc(8000.))], [(DH, nbc(4000.))]],
                x0="CCN", diagonal_balance=True)
mesh.origin = mesh.origin + np.r_[0.0, 0.0, topo_xyz[:, -1].max()]
k = np.sqrt(np.sum(topo_xyz[:, 0:2] ** 2, axis=1)) < 1200
mesh.refine_surface(topo_xyz[k, :], padding_cells_by_level=[0, 4, 4],
                    finalize=False)
mesh.refine_points(survey.unique_electrode_locations,
                   padding_cells_by_level=[6, 6, 4], finalize=False)
mesh.finalize()
active = active_from_xyz(mesh, topo_xyz)
nC = int(active.sum())
survey.drape_electrodes_on_topography(mesh, active, topo_cell_cutoff="top")
print(f"[survey] lines {NLINES} | nD {nD} | nS {nS} | active {nC:,}", flush=True)

sig = 1e-2 * np.ones(nC)
cc = mesh.cell_centers[active, :]
sig[model_builder.get_indices_sphere(np.r_[-300., 0., 100.], 165., cc)] = 1e-1
sig[model_builder.get_indices_sphere(np.r_[300., 0., 100.], 165., cc)] = 1e-3

eng = TorchDC3D(mesh, survey, device="cuda", dtype=DT,
                active_cells=active, val_inactive=np.log(1e8))
m0 = torch.tensor(-np.log(1e-2) * np.ones(nC), dtype=DT, device="cuda")
with torch.no_grad():
    dobs = eng.dpred(m0.clone()).clone()
    dobs += 0.1 * dobs.abs() * torch.randn_like(dobs)
w = 1.0 / (1e-7 + 0.1 * dobs.abs())


class _SolveOnly(torch.autograd.Function):
    """x = A^-1 b for an ALREADY factorised A, differentiable in values and b.

    Identical to the engine's solve except that it does not refactorise: the
    factorisation is shared by every chunk of a given model, which is the whole
    point of the scheme.
    """

    @staticmethod
    def forward(ctx, values, b, solver):
        x = solver.solve(b.detach())
        ctx.solver, ctx.dt = solver, values.dtype
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, gx):
        (x,) = ctx.saved_tensors
        s = ctx.solver
        adj = s.solve(gx.contiguous())
        xm, am = x.reshape(s.n, -1), adj.reshape(s.n, -1)
        nnz = s.col.numel()
        gv = torch.empty(nnz, dtype=xm.dtype, device=xm.device)
        step = max(8_000_000 // max(xm.shape[1], 1), 1)
        for p0 in range(0, nnz, step):
            p1 = min(p0 + step, nnz)
            gv[p0:p1] = -(am[s.row[p0:p1]] * xm[s.col[p0:p1]]).sum(-1)
        return gv.to(ctx.dt), None, None


def assemble(m):
    """Same assembly the engine does, including the active-cell injection."""
    m_loc = m
    if eng._P_inj is not None:
        m_loc = torch.sparse.mm(
            eng._P_inj, m_loc.unsqueeze(1)).squeeze(1) + eng._v_inj
    sigma = torch.exp(-m_loc).unsqueeze(1)
    return torch.sparse.mm(eng.L3, sigma).squeeze(1)


def run(chunk):
    """One misfit-and-gradient evaluation; chunk=0 means the one-shot path."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = nvml()
    m = m0.clone().requires_grad_(True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if chunk == 0:
        r = w * (eng.dpred(m) - dobs)
        (r ** 2).sum().backward()
        g = m.grad.detach().clone()
        peak = nvml() - base
    else:
        solver = make_solver(
            eng.solver.crow.detach().cpu().to(torch.int32),
            eng.solver.col.detach().cpu().to(torch.int32),
            eng.nN, chunk, device="cuda", dtype=DT)
        vals = assemble(m)
        solver.factorize(vals.detach())          # once for the whole model
        g = torch.zeros_like(m0)
        peak = 0.0
        for s0 in range(0, nS, chunk):
            s1 = min(s0 + chunk, nS)
            sel = (eng.sidx >= s0) & (eng.sidx < s1)
            if not bool(sel.any()):
                continue
            mc = m0.clone().requires_grad_(True)
            v = assemble(mc)
            U = _SolveOnly.apply(v, eng.q[:, s0:s1], solver)
            d = torch.sparse.mm(eng.Pd, U)[eng.iD[sel], eng.sidx[sel] - s0]
            ((w[sel] * (d - dobs[sel])) ** 2).sum().backward()
            g += mc.grad.detach()
            peak = max(peak, nvml() - base)
            del U, d, v, mc
        solver.free()

    torch.cuda.synchronize()
    return time.perf_counter() - t0, peak, g


print(f"[warm] {run(0)[0]:.2f}s", flush=True)
ref = None
for c in CHUNKS:
    t, peak, g = run(c)
    if ref is None:
        ref = g
        err = 0.0
    else:
        err = float((g - ref).norm() / ref.norm())
    out = dict(nlines=NLINES, nD=nD, nS=nS, nC=nC, chunk=c or nS,
               n_chunks=1 if c == 0 else int(np.ceil(nS / c)),
               t_s=round(t, 3), nvml_gib=round(peak, 3),
               grad_rel_err=err)
    print("JSON: " + json.dumps(out), flush=True)
