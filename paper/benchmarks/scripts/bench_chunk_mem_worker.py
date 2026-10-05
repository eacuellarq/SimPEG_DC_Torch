"""ONE point of the source-chunking memory study, in its own process.

Why a separate process per point: cuDSS holds device memory outside PyTorch's
caching allocator, so `empty_cache()` does not return it and a second
configuration measured in the same process reads a peak that is really the
first configuration's. Every number here is therefore a cold process.

Measures, at a given chunk width and for a given mode:
  MODE=fwd   forward only  — fields are projected to data and released
  MODE=grad  forward + gradient — fields must be retained for the adjoint

The difference between the two is the cost of differentiability; the change
with the chunk width is what chunking actually buys.

Env: NLINES (def 65), CHUNK (0 = all sources at once), MODE (fwd|grad),
     PREC (f32|f64). Emits one "JSON: {...}" line.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import threading
import time

import numpy as np
import torch  # noqa: F401  (before simpeg/pydiso: MKL DLL clash)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

import pynvml

pynvml.nvmlInit()
_h = pynvml.nvmlDeviceGetHandleByIndex(0)
_peak, _stop = [0.0], [False]


def _sample():
    while not _stop[0]:
        _peak[0] = max(_peak[0],
                       pynvml.nvmlDeviceGetMemoryInfo(_h).used / 2 ** 30)
        time.sleep(0.05)


threading.Thread(target=_sample, daemon=True).start()

from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line)
from discretize import TreeMesh
from discretize.utils import active_from_xyz

from dctorch import TorchDC3D
from dctorch.solver import make_solver

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "..", "..", "..", "examples",
                   "inv_dcr_3d_files") + os.path.sep
DH = 25.0
NLINES = int(os.environ.get("NLINES", "65"))
CHUNK = int(os.environ.get("CHUNK", "0"))
MODE = os.environ.get("MODE", "grad")
DT = {"f32": torch.float32, "f64": torch.float64}[os.environ.get("PREC", "f32")]
np.random.seed(0)

topo = np.loadtxt(DIR + "topo_xyz.txt")
srcs = []
for y in np.linspace(-800, 800, NLINES):
    srcs += generate_dcip_sources_line(
        "dipole-dipole", "volt", "3D", np.r_[-1000.0, 1000.0, y, y], topo,
        num_rx_per_src=8, station_spacing=50.0)
survey = dc.Survey(srcs)
nS, nD = len(survey.source_list), int(survey.nD)

nbc = lambda w: 2 ** int(np.round(np.log(w / DH) / np.log(2.0)))  # noqa: E731
mesh = TreeMesh([[(DH, nbc(8000.))], [(DH, nbc(8000.))], [(DH, nbc(4000.))]],
                x0="CCN", diagonal_balance=True)
mesh.origin = mesh.origin + np.r_[0.0, 0.0, topo[:, -1].max()]
k = np.sqrt(np.sum(topo[:, 0:2] ** 2, axis=1)) < 1200
mesh.refine_surface(topo[k, :], padding_cells_by_level=[0, 4, 4],
                    finalize=False)
mesh.refine_points(survey.unique_electrode_locations,
                   padding_cells_by_level=[6, 6, 4], finalize=False)
mesh.finalize()
active = active_from_xyz(mesh, topo)
nC = int(active.sum())
survey.drape_electrodes_on_topography(mesh, active, topo_cell_cutoff="top")

eng = TorchDC3D(mesh, survey, device="cuda", dtype=DT,
                active_cells=active, val_inactive=np.log(1e8))
m0 = torch.tensor(-np.log(1e-2) * np.ones(nC), dtype=DT, device="cuda")
with torch.no_grad():
    dobs = eng.dpred(m0.clone()).clone()
    dobs += 0.1 * dobs.abs() * torch.randn_like(dobs)
w = 1.0 / (1e-7 + 0.1 * dobs.abs())

# baseline: everything that exists before the evaluation under test
torch.cuda.empty_cache()
time.sleep(0.4)
base = _peak[0] = pynvml.nvmlDeviceGetMemoryInfo(_h).used / 2 ** 30


class _SolveOnly(torch.autograd.Function):
    """A^-1 b for an already factorised A; the factor is shared by all chunks."""

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
    mi = m
    if eng._P_inj is not None:
        mi = torch.sparse.mm(eng._P_inj, mi.unsqueeze(1)).squeeze(1) + eng._v_inj
    return torch.sparse.mm(eng.L3, torch.exp(-mi).unsqueeze(1)).squeeze(1)


torch.cuda.synchronize()
t0 = time.perf_counter()

if CHUNK == 0:
    if MODE == "fwd":
        with torch.no_grad():
            d = eng.dpred(m0)
        chk = float(d.abs().sum())
    else:
        m = m0.clone().requires_grad_(True)
        ((w * (eng.dpred(m) - dobs)) ** 2).sum().backward()
        chk = float(m.grad.abs().sum())
else:
    crow = eng.solver.crow.detach().cpu().to(torch.int32)
    col = eng.solver.col.detach().cpu().to(torch.int32)
    eng.solver.free()                      # only the chunk solver should live
    torch.cuda.empty_cache()
    slv = make_solver(crow, col, eng.nN, CHUNK, device="cuda", dtype=DT)
    slv.factorize(assemble(m0).detach())   # once, shared by every chunk
    g = torch.zeros_like(m0)
    acc = 0.0
    for s0 in range(0, nS, CHUNK):
        s1 = min(s0 + CHUNK, nS)
        sel = (eng.sidx >= s0) & (eng.sidx < s1)
        if not bool(sel.any()):
            continue
        if MODE == "fwd":
            with torch.no_grad():
                U = slv.solve(eng.q[:, s0:s1].contiguous())
                d = torch.sparse.mm(eng.Pd, U)[eng.iD[sel], eng.sidx[sel] - s0]
            acc += float(d.abs().sum())
        else:
            mc = m0.clone().requires_grad_(True)
            U = _SolveOnly.apply(assemble(mc), eng.q[:, s0:s1].contiguous(), slv)
            d = torch.sparse.mm(eng.Pd, U)[eng.iD[sel], eng.sidx[sel] - s0]
            ((w[sel] * (d - dobs[sel])) ** 2).sum().backward()
            g += mc.grad.detach()
            del mc
        del U, d
    chk = acc if MODE == "fwd" else float(g.abs().sum())

torch.cuda.synchronize()
t = time.perf_counter() - t0
time.sleep(0.3)
_stop[0] = True

print("JSON: " + json.dumps(dict(
    nlines=NLINES, nD=nD, nS=nS, nC=nC, nN=int(eng.nN),
    prec=str(DT).split(".")[-1], mode=MODE, chunk=CHUNK or nS,
    t_s=round(t, 3), nvml_peak_gib=round(_peak[0], 3),
    nvml_base_gib=round(base, 3),
    nvml_delta_gib=round(_peak[0] - base, 3), check=chk)), flush=True)
