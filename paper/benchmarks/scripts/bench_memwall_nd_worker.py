"""
bench_memwall_nd_worker.py — UN punto de la pared de memoria vs TAMANO DEL
SURVEY (N_d) a malla fija (dos-esferas dh=25, publico): se densifica el
survey dipolo-dipolo sobre el MISMO modelo verdadero (NLINES lineas E-W,
DEC decimacion de estaciones), dato sintetico 10% ruido, y se mide:

  ENGINE=gn      : SimPEG GN canonica storeJ=True, MAXITER=2 -> RAM RSS pico
  ENGINE=dctorch : TorchDC3D F32, 1 closure fwd+bwd warm -> NVML pico

Env: NLINES, DEC, ENGINE. Emite "JSON: {...}".
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import time

import numpy as np
import torch  # noqa: F401  (antes que simpeg/pydiso: conflicto de DLLs MKL)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from simpeg import maps
from simpeg.electromagnetics.static import resistivity as dc
from simpeg.electromagnetics.static.utils.static_utils import (
    generate_dcip_sources_line)
from simpeg.utils import model_builder
from discretize import TreeMesh
from discretize.utils import active_from_xyz

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "inv_dcr_3d_files") + os.path.sep
DH = 25.0
NLINES = int(os.environ.get("NLINES", "9"))
DEC = int(os.environ.get("DEC", "1"))       # 1 = estaciones cada 50 m
ENGINE = os.environ.get("ENGINE", "dctorch")
np.random.seed(0)

topo_xyz = np.loadtxt(DIR + "topo_xyz.txt")

# ---------- survey denso: NLINES lineas E-W sobre el nucleo ----------
ys = np.linspace(-800, 800, NLINES)
srcs = []
for y in ys:
    end_pts = np.r_[-1000.0, 1000.0, y, y]      # [x1, x2, y1, y2]
    srcs += generate_dcip_sources_line(
        "dipole-dipole", "volt", "3D", end_pts, topo_xyz,
        num_rx_per_src=8, station_spacing=50.0 * DEC)
survey = dc.Survey(srcs)
nD, nS = int(survey.nD), len(survey.source_list)

# ---------- malla del tutorial (dh=25) ----------
nbc = lambda w: 2 ** int(np.round(np.log(w / DH) / np.log(2.0)))  # noqa
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
print(f"[survey] lines {NLINES} | nD {nD} | nS {nS} | active {nC:,}",
      flush=True)

# ---------- modelo verdadero (esferas del tutorial) ----------
sig = 1e-2 * np.ones(nC)
cc = mesh.cell_centers[active, :]
sig[model_builder.get_indices_sphere(np.r_[-300., 0., 100.], 165., cc)] = 1e-1
sig[model_builder.get_indices_sphere(np.r_[300., 0., 100.], 165., cc)] = 1e-3
m_true = np.log(sig)

if ENGINE == "gn":
    import psutil
    from simpeg import (data, data_misfit, directives, inverse_problem,
                        inversion, optimization, regularization)
    from pymatsolver import Pardiso
    smap = maps.InjectActiveCells(mesh, active, 1e-8) * maps.ExpMap(nP=nC)
    sim = dc.simulation.Simulation3DNodal(mesh, survey=survey, sigmaMap=smap,
                                          storeJ=True, solver=Pardiso)
    d_clean = sim.dpred(m_true)
    dobs = d_clean + 0.1 * np.abs(d_clean) * np.random.randn(nD)
    std = 1e-7 + 0.1 * np.abs(dobs)
    dat = data.Data(survey, dobs=dobs, standard_deviation=std)
    dmis = data_misfit.L2DataMisfit(data=dat, simulation=sim)
    m0 = np.log(1e-2) * np.ones(nC)
    reg = regularization.WeightedLeastSquares(
        mesh, active_cells=active, length_scale_x=100.0,
        length_scale_y=100.0, length_scale_z=100.0, reference_model=m0)
    opt = optimization.InexactGaussNewton(maxIter=2, maxIterLS=20,
                                          cg_maxiter=30, cg_rtol=1e-3)
    invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)
    dirs = [directives.UpdateSensitivityWeights(every_iteration=True,
                                                threshold_value=1e-2),
            directives.UpdatePreconditioner(update_every_iteration=True),
            directives.BetaEstimate_ByEig(beta0_ratio=100, random_seed=0),
            directives.BetaSchedule(coolingFactor=2.0, coolingRate=2)]
    t0 = time.time()
    inversion.BaseInversion(invProb, directiveList=dirs).run(m0.copy())
    rss = psutil.Process().memory_info().peak_wset / 2**30
    print("JSON: " + json.dumps(dict(
        engine="gn-storeJ", nD=nD, nS=nS, nC=nC, nlines=NLINES,
        mem_gib=round(rss, 2), t=round(time.time() - t0, 1))), flush=True)
else:
    import threading
    import pynvml
    from dctorch import TorchDC3D
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    peak, stop = [0.0], [False]

    def samp():
        while not stop[0]:
            peak[0] = max(peak[0],
                          pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30)
            time.sleep(0.05)

    threading.Thread(target=samp, daemon=True).start()
    eng = TorchDC3D(mesh, survey, device="cuda", dtype=torch.float32,
                    active_cells=active, val_inactive=np.log(1e8))
    mt_true = torch.tensor(-m_true, dtype=torch.float64, device="cuda")
    with torch.no_grad():
        d_clean = eng.dpred(mt_true)
    dobs_t = d_clean + 0.1 * d_clean.abs() * torch.randn(
        nD, dtype=torch.float64, device="cuda")
    W_t = 1.0 / (1e-7 + 0.1 * dobs_t.abs())
    m = torch.tensor(np.log(100.0) * np.ones(nC), dtype=torch.float64,
                     device="cuda", requires_grad=True)
    t0 = time.time()
    for _ in range(2):                       # warm: plan+factor y closure
        m.grad = None
        d = eng.dpred(m).to(torch.float64)
        loss = 0.5 * ((W_t * (d - dobs_t)) ** 2).sum()
        loss.backward()
        torch.cuda.synchronize()
    time.sleep(0.2)
    stop[0] = True
    print("JSON: " + json.dumps(dict(
        engine="dctorch-f32", nD=nD, nS=nS, nC=nC, nlines=NLINES,
        mem_gib=round(peak[0], 2), t=round(time.time() - t0, 1))),
        flush=True)
