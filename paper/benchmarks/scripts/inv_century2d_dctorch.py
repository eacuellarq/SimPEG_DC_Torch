"""
inv_century2d_dctorch.py — Century 2D (misma linea 46800E) con dctorch
TorchDC2D: MISMO survey/malla/std/arranque/reg/schedule/target que el vanilla
(century_common). Modelo log-rho en TODA la malla (sin topo). Convencion ½:
target = nD/2. Diferencias declaradas: L-BFGS(20)/nivel, beta0 por ratio.

Env: DCTORCH_PREC=f32|f64, MAXOUTER (def 40).
Salida: results de paper/benchmarks + JSON. Correr bajo watchdog.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from simpeg import regularization

from century_common import (REPO, BETA0_RATIO, COOL, COOLING_RATE, ALPHA_X,
                            build_mesh, build_survey, starting_halfspace)
from dctorch import TorchDC2D

PREC = os.environ.get("DCTORCH_PREC", "f32")
DT = {"f64": torch.float64, "f32": torch.float32}[PREC]
MAX_OUTER = int(os.environ.get("MAXOUTER", "40"))
LBFGS_MAXITER = 20
torch.manual_seed(0)
np.random.seed(0)

survey, dobs, std = build_survey()
mesh, dx = build_mesh(survey)
nD = int(survey.nD)
nC = int(mesh.nC)
rho0 = starting_halfspace(survey, dobs)
print(f"[setup] nD {nD} | {nC:,} celdas | dx {dx:g} | rho0 {rho0:.1f}",
      flush=True)

t0 = time.time()
eng = TorchDC2D(mesh, survey, device="cuda", dtype=DT)
t_build = time.time() - t0
print(f"[build] {t_build:.1f}s | precision {PREC}", flush=True)

mref = np.log(rho0) * np.ones(nC)
reg = regularization.WeightedLeastSquares(
    mesh, alpha_s=1.0 / dx ** 2, alpha_x=ALPHA_X, alpha_y=ALPHA_X,
    reference_model=mref)
R = sp.csr_matrix(reg.deriv2(mref)).tocoo()
R_t = torch.sparse_coo_tensor(
    torch.tensor(np.vstack((R.row, R.col)), dtype=torch.int64, device="cuda"),
    torch.tensor(R.data, dtype=torch.float64, device="cuda"),
    R.shape).coalesce()
mref_t = torch.tensor(mref, dtype=torch.float64, device="cuda")
W_t = torch.tensor(1.0 / std, dtype=torch.float64, device="cuda")
dobs_t = torch.tensor(dobs, dtype=torch.float64, device="cuda")
LOB, HIB = float(np.log(1e-2)), float(np.log(1e6))   # anti-overflow, ancho


def clamp(m):
    return torch.clamp(m, LOB, HIB)


def phi_m(mc):
    d2 = (mc - mref_t).unsqueeze(1)
    return 0.5 * torch.sum(d2 * torch.sparse.mm(R_t, d2))


def phi_d(mc):
    d = eng.dpred(mc).to(torch.float64)
    return 0.5 * torch.sum((W_t * (d - dobs_t)) ** 2)


target = 0.5 * nD
m = torch.tensor(mref.copy(), dtype=torch.float64, device="cuda",
                 requires_grad=True)
with torch.no_grad():
    phid0 = float(phi_d(mref_t))
    pm0 = float(phi_m(clamp(mref_t + 0.1 * torch.randn(
        nC, dtype=torch.float64, device="cuda"))))
beta = BETA0_RATIO * phid0 / (pm0 + 1e-30)
print(f"[init] phi_d0 {phid0:.3e} | beta0 {beta:.3e} | target(½) {target:.0f}",
      flush=True)

opt = torch.optim.LBFGS([m], lr=1.0, max_iter=LBFGS_MAXITER, history_size=10,
                        line_search_fn="strong_wolfe")
torch.cuda.reset_peak_memory_stats()
hist = []
t0 = time.time()
for outer in range(MAX_OUTER):
    pd_, pm_ = [None], [None]
    tA = time.time()

    def closure():
        opt.zero_grad()
        mc = clamp(m)
        fd = phi_d(mc)
        fm = phi_m(mc)
        loss = fd + beta * fm
        loss.backward()
        pd_[0], pm_[0] = float(fd), float(fm)
        return loss

    opt.step(closure)
    hist.append((pd_[0], pm_[0], beta, time.time() - tA))
    print(f"{outer+1:4d} {pd_[0]:12.3e} {pm_[0]:12.3e} {beta:11.3e} "
          f"{time.time()-tA:6.1f}", flush=True)
    if pd_[0] <= target:
        print(f"*** TARGET: phi_d {pd_[0]:.1f} <= {target:.0f} ***")
        break
    if (outer + 1) % COOLING_RATE == 0:
        beta /= COOL

t_loop = time.time() - t0
vram = torch.cuda.max_memory_reserved() / 2**30
chi = 2 * hist[-1][0] / nD
m_np = clamp(m).detach().cpu().numpy()
out = os.path.join(REPO, "results", f"inv_century2d_dctorch_{PREC}.npz")
np.savez_compressed(out, m_log_rho=m_np, hist=np.array(hist), t=t_loop,
                    t_build=t_build, chi=chi, nC=nC, nD=nD, dx=dx, rho0=rho0,
                    prec=PREC)
print(f"\n[century-dctorch-{PREC}] loop {t_loop:.1f}s (build {t_build:.1f}s) "
      f"| {len(hist)} outers | chi {chi:.3f} | VRAM {vram:.2f} GiB",
      flush=True)
print("JSON: " + json.dumps(dict(engine=f"century-dctorch-{PREC}",
                                 t=round(t_loop, 1),
                                 t_build=round(t_build, 1), iters=len(hist),
                                 chi=round(chi, 3), nC=nC, nD=nD,
                                 vram_gib=round(vram, 2))))
print(f"saved {out}", flush=True)
