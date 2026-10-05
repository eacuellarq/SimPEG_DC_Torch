"""
inv_century_ip_dctorch.py — inversion de cargabilidad LINEALIZADA (Seigel)
de la linea Century 46800E con dctorch TorchIP2D. Fondo: sigma=exp(-m_dc)
con m_dc = modelo DC dctorch F64 (chi 0.965) — MISMO fondo que el baseline
vanilla (diferencia declarada vs tutorial, que usa su propio DC) para que
ambos motores inviertan el MISMO operador lineal. Receta IP del tutorial
Transform-2020: eta0=1e-3, alpha_s=1/dx², cooling 2 cada 1, chifact 1,
cotas [0, 1000] mV/V (eta<1 V/V; tutorial usa upper=inf — declarado).
Convencion ½: target = nD/2. eta en mV/V (unidad del dato, operador lineal).

Env: DCTORCH_PREC=f64|f32 (def f64), MAXOUTER (def 40).
Correr (CWD scripts/, env simpeg311): python -u inv_century_ip_dctorch.py
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

from century_common import REPO, ALPHA_X
from century_ip_common import (COOL_IP, COOLING_RATE_IP, ETA0,
                               build_dc_survey_and_mesh, build_ip_survey,
                               load_background)
from dctorch import TorchIP2D

PREC = os.environ.get("DCTORCH_PREC", "f64")
DT = {"f64": torch.float64, "f32": torch.float32}[PREC]
MAX_OUTER = int(os.environ.get("MAXOUTER", "40"))
LBFGS_MAXITER = 20
torch.manual_seed(0)
np.random.seed(0)

_, mesh, dx = build_dc_survey_and_mesh()
m_dc, chi_dc = load_background(mesh)
survey, dobs, std = build_ip_survey()
nD, nC = int(survey.nD), int(mesh.nC)
print(f"[setup] IP nD {nD} | {nC:,} celdas | dx {dx:g} | fondo DC chi "
      f"{chi_dc:.3f}", flush=True)

t0 = time.time()
eng = TorchIP2D(mesh, survey, m_dc, device="cuda", dtype=DT)
t_build = time.time() - t0
print(f"[build] {t_build:.1f}s | precision {PREC}", flush=True)

mref = ETA0 * np.ones(nC)
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
LOB, HIB = 0.0, 1000.0                     # eta en mV/V: [0, 1 V/V)


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
# beta0 estilo BetaEstimate_ByEig: cociente de Rayleigh con x aleatorio.
# El operador IP es LINEAL => J x = dpred(x) exacto (sin FD ni autograd).
with torch.no_grad():
    phid0 = float(phi_d(mref_t))
    x = torch.randn(nC, dtype=torch.float64, device="cuda")
    num = float(torch.sum((W_t * eng.dpred(x).to(torch.float64)) ** 2))
    den = float(torch.sum(x.unsqueeze(1)
                          * torch.sparse.mm(R_t, x.unsqueeze(1))))
beta = num / den
print(f"[init] phi_d0 {phid0:.3e} | beta0 {beta:.3e} (eig-ratio) | "
      f"target(½) {target:.0f}", flush=True)

# sub-pasos de 5 iters con chequeo de target entre ellos (mismo presupuesto
# 20/level que el DC, pero la parada no se pasa del target medio nivel:
# el problema IP es lineal y L-BFGS converge el nivel completo de golpe)
SUB = 5
opt = torch.optim.LBFGS([m], lr=1.0, max_iter=SUB, history_size=10,
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

    for _ in range(LBFGS_MAXITER // SUB):
        opt.step(closure)
        # proyeccion del iterando: en IP la cota eta>=0 es ACTIVA en media
        # malla; sin esto el m crudo deriva bajo cero con gradiente nulo
        # (el clamp del closure solo protege la fisica, no al optimizador)
        with torch.no_grad():
            m.clamp_(LOB, HIB)
        if pd_[0] <= target:
            break
    hist.append((pd_[0], pm_[0], beta, time.time() - tA))
    print(f"{outer+1:4d} {pd_[0]:12.3e} {pm_[0]:12.3e} {beta:11.3e} "
          f"{time.time()-tA:6.1f}", flush=True)
    if pd_[0] <= target:
        print(f"*** TARGET: phi_d {pd_[0]:.1f} <= {target:.0f} ***")
        break
    if (outer + 1) % COOLING_RATE_IP == 0:
        beta /= COOL_IP

t_loop = time.time() - t0
vram = torch.cuda.max_memory_reserved() / 2**30
chi = 2 * hist[-1][0] / nD
eta_np = clamp(m).detach().cpu().numpy()
out = os.path.join(REPO, "results", f"inv_century_ip_dctorch_{PREC}.npz")
np.savez_compressed(out, eta=eta_np, hist=np.array(hist), t=t_loop,
                    t_build=t_build, chi=chi, nC=nC, nD=nD, dx=dx,
                    chi_dc=chi_dc, prec=PREC)
print(f"\n[century-ip-dctorch-{PREC}] loop {t_loop:.1f}s (build {t_build:.1f}s)"
      f" | {len(hist)} outers | chi {chi:.3f} | VRAM {vram:.2f} GiB",
      flush=True)
print("JSON: " + json.dumps(dict(engine=f"century-ip-dctorch-{PREC}",
                                 t=round(t_loop, 1),
                                 t_build=round(t_build, 1), iters=len(hist),
                                 chi=round(chi, 3), nC=nC, nD=nD,
                                 vram_gib=round(vram, 2))))
print(f"saved {out}", flush=True)
