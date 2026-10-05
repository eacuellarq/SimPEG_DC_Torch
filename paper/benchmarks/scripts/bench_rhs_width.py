"""Is a wide right-hand-side block actually faster to solve?

Isolates the triangular solve from the autograd graph and from the gradient
contraction: one factorisation, then the same 2470 right-hand sides covered in
blocks of width k, timing only `solver.solve`. If the total time is flat or
falls as k shrinks, then the wide block buys nothing and the memory it costs is
pure loss.

Env: NLINES (def 65), WIDTHS (def "2470,1024,512,256,128,64,32").
CWD scripts/, env simpeg311.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time

import numpy as np
import torch  # noqa: F401

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line)
from discretize import TreeMesh
from discretize.utils import active_from_xyz
import pynvml

from dctorch import TorchDC3D
from dctorch.solver import make_solver

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "..", "..", "..", "examples",
                   "inv_dcr_3d_files") + os.path.sep
DH = 25.0
NLINES = int(os.environ.get("NLINES", "65"))
WIDTHS = [int(w) for w in
          os.environ.get("WIDTHS", "2470,1024,512,256,128,64,32").split(",")]
DT = torch.float32
np.random.seed(0)

pynvml.nvmlInit()
_h = pynvml.nvmlDeviceGetHandleByIndex(0)
nvml = lambda: pynvml.nvmlDeviceGetMemoryInfo(_h).used / 2 ** 30  # noqa: E731

topo = np.loadtxt(DIR + "topo_xyz.txt")
srcs = []
for y in np.linspace(-800, 800, NLINES):
    srcs += generate_dcip_sources_line(
        "dipole-dipole", "volt", "3D", np.r_[-1000.0, 1000.0, y, y], topo,
        num_rx_per_src=8, station_spacing=50.0)
survey = dc.Survey(srcs)
nS = len(survey.source_list)

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
survey.drape_electrodes_on_topography(mesh, active, topo_cell_cutoff="top")

eng = TorchDC3D(mesh, survey, device="cuda", dtype=DT,
                active_cells=active, val_inactive=np.log(1e8))
nN = eng.nN
m = torch.tensor(-np.log(1e-2) * np.ones(int(active.sum())), dtype=DT,
                 device="cuda")
mi = torch.sparse.mm(eng._P_inj, m.unsqueeze(1)).squeeze(1) + eng._v_inj
vals = torch.sparse.mm(eng.L3, torch.exp(-mi).unsqueeze(1)).squeeze(1)

# capture the pattern BEFORE releasing the engine's solver, then free it so
# the widest block does not coexist with a second factorisation of its size
crow = eng.solver.crow.detach().cpu().to(torch.int32)
col = eng.solver.col.detach().cpu().to(torch.int32)
eng.solver.free()
torch.cuda.empty_cache()
print(f"[setup] nN {nN:,} | nS {nS} | dtype {DT}", flush=True)
print(f"{'width':>6} {'blocks':>7} {'total s':>9} {'ms/RHS':>8} "
      f"{'NVML GiB':>9}", flush=True)

ref = None
for w in WIDTHS:
    torch.cuda.empty_cache()
    base = nvml()
    slv = make_solver(crow, col, nN, w, device="cuda", dtype=DT)
    slv.factorize(vals)
    b = eng.q[:, :w].contiguous()
    for _ in range(2):                       # warm
        slv.solve(b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    acc = 0.0
    for s0 in range(0, nS, w):
        s1 = min(s0 + w, nS)
        bb = eng.q[:, s0:s1].contiguous()
        x = slv.solve(bb)
        acc += float(x[0, 0])                # keep it honest, force the result
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    peak = nvml() - base
    nb = int(np.ceil(nS / w))
    print(f"{w:>6} {nb:>7} {t:>9.3f} {t / nS * 1e3:>8.3f} {peak:>9.3f}",
          flush=True)
    slv.free()
    del slv
