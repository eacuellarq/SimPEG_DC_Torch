"""Por que no hay victoria aplastante en Century: sonda de tamano.
Mide forward+grad por evaluacion vanilla-vs-dctorch en la malla del
tutorial (dx=25) y en una fina (dx=10, ~1/10 spacing como el tutorial 2D
moderno), e imprime nky de ambos motores."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from simpeg import maps
from simpeg.electromagnetics.static import resistivity as dc
from pymatsolver import Pardiso
import century_common as cc
from dctorch import TorchDC2D

survey, dobs, std = cc.build_survey()
W = 1.0 / std

for div in (4, 10):
    # malla: dx = spacing/div
    import importlib
    import century_common
    mesh, dx = None, None
    # reconstruir con dx custom
    import discretize
    locs = survey.unique_electrode_locations
    spac = np.abs(np.diff(np.unique(locs[:, 0]))).min()
    dxv = spac / div
    x0, x1 = locs[:, 0].min(), locs[:, 0].max()
    mid_ab = (survey.locations_a + survey.locations_b) / 2
    mid_mn = (survey.locations_m + survey.locations_n) / 2
    max_sep = float(np.abs(mid_ab - mid_mn)[:, 0].max())
    n_core_x = int(np.ceil((x1 - x0) / dxv)) + 8
    n_core_z = int(np.ceil(max_sep / 3.0 / dxv))
    n_pad = 1
    while dxv * (1.3 ** np.arange(1, n_pad + 1)).sum() < 3 * max_sep:
        n_pad += 1
    mesh = discretize.TensorMesh(
        [[(dxv, n_pad, -1.3), (dxv, n_core_x), (dxv, n_pad, 1.3)],
         [(dxv, n_pad, -1.3), (dxv, n_core_z)]])
    mesh.origin = np.r_[x0 - 4 * dxv - mesh.h[0][:n_pad].sum(),
                        -mesh.h[1].sum()]

    # --- vanilla: fields+dpred y Jtvec ---
    simv = dc.simulation_2d.Simulation2DNodal(
        mesh, survey=survey, rhoMap=maps.ExpMap(mesh), storeJ=False,
        solver=Pardiso)
    m = np.log(100.0) * np.ones(mesh.nC)
    f = simv.fields(m); d = simv.dpred(m, f=f)   # warm
    tf, tg = [], []
    for _ in range(3):
        m2 = m + np.random.randn(mesh.nC) * 1e-3
        t0 = time.time(); f = simv.fields(m2); d = simv.dpred(m2, f=f)
        tf.append(time.time() - t0)
        v = W * (W * (d - dobs))
        t0 = time.time(); g = simv.Jtvec(m2, v, f=f)
        tg.append(time.time() - t0)
    nky_v = len(simv._quad_points)

    # --- dctorch F32 ---
    eng = TorchDC2D(mesh, survey, device="cuda", dtype=torch.float32)
    mt = torch.tensor(m, dtype=torch.float64, device="cuda",
                      requires_grad=True)
    Wt = torch.tensor(W, device="cuda"); dt_ = torch.tensor(dobs, device="cuda")
    def closure():
        mt.grad = None
        dd = eng.dpred(mt).to(torch.float64)
        loss = 0.5 * ((Wt * (dd - dt_)) ** 2).sum()
        loss.backward(); torch.cuda.synchronize()
    closure()  # warm (plan+factor)
    tt = []
    for _ in range(5):
        t0 = time.time(); closure(); tt.append(time.time() - t0)
    print(f"dx={dxv:g} ({mesh.nC:,} celdas, nky {nky_v}/{eng.nky}): "
          f"vanilla fwd {np.median(tf)*1e3:.0f}ms + Jtvec "
          f"{np.median(tg)*1e3:.0f}ms | dctorch fwd+bwd "
          f"{np.median(tt)*1e3:.0f}ms | ratio "
          f"{(np.median(tf)+np.median(tg))/np.median(tt):.1f}x", flush=True)
