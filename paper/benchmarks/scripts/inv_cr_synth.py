"""
inv_cr_synth.py — la TERCERA formulacion sobre el MISMO dato sintetico:
inversion de RESISTIVIDAD COMPLEJA (Kemna 2000 / RES2DINV / cR2), y
comparacion a tres bandas contra el secuencial linealizado y el conjunto
de dos estados (que ya corrieron en inv_dcip_synth.py).

El terreno comun para comparar los tres es la resistividad DC (polarizada):
  - secuencial : rho_seq ES rho_0 (ajusta d_DC directamente)
  - dos estados: rho_0 = 1/(sigma*(1-eta)) del modelo recuperado
  - complejo   : rho_DC es el parametro directamente
y la cargabilidad (eta ~ m). Asi se evita la reconstruccion sigma_inf =
sigma_0/(1-eta), que introduce error propio al dividir dos campos suavizados.

Env: MAXOUTER (def 40) | ALPHA_M (def 1) | LAM (def 1) | TAG
Correr (CWD scripts/, env simpeg311): python -u inv_cr_synth.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from dcip_joint import beta0_blocks, make_reg, quad
from dcip_synth_common import CHRG, REPO, make_data, starting_halfspace
from dctorch import TorchCR2D

MAX_OUTER = int(os.environ.get("MAXOUTER", "40"))
ALPHA_M = float(os.environ.get("ALPHA_M", "1.0"))
LAM = float(os.environ.get("LAM", "1.0"))
TAG = os.environ.get("TAG", "")
M0 = float(os.environ.get("M0", "0.05"))
DEV = "cuda"
torch.manual_seed(0)
np.random.seed(0)

t00 = time.time()
D = make_data(device=DEV, dc_state="total")
mesh, dx, survey, nD = D["mesh"], D["dx"], D["survey"], D["nD"]
nC = int(mesh.nC)
rho0 = starting_halfspace(survey, D["dobs_dc"])
print(f"[setup] nD {nD} x2 | {nC:,} celdas ({2*nC:,} parametros) | dx {dx:g} "
      f"| rho0 {rho0:.1f} | eta_true max {D['eta_true'].max():.2f}", flush=True)

t0 = time.time()
eng = TorchCR2D(mesh, survey, device=DEV, dtype=torch.complex128)
t_build = time.time() - t0
print(f"[build] {t_build:.1f}s | solver COMPLEJO SIMETRICO", flush=True)

# El dato DC de este survey es NEGATIVO en los 140 cuadripolos (convencion de
# polaridad M-N). En la formulacion compleja el observable emparejado con la
# fase es la MAGNITUD |V*| (de ahi sale la resistividad aparente), asi que se
# compara |V*| contra |V_obs|; el signo es convencion de geometria y la fase
# Im/Re es invariante a el. Emparejar |V*| con el dato con signo deja chi2_DC
# clavado en ~2500 (medido).
W_r = torch.tensor(1.0 / D["std_dc"], dtype=torch.float64, device=DEV)
W_i = torch.tensor(1.0 / D["std_ip"], dtype=torch.float64, device=DEV)
O_r = torch.tensor(np.abs(D["dobs_dc"]), dtype=torch.float64, device=DEV)
O_i = torch.tensor(D["dobs_ip"], dtype=torch.float64, device=DEV)

mref_r = np.log(rho0) * np.ones(nC)
mref_m = M0 * np.ones(nC)                 # SIN transformacion, SIN cotas
R_r, Rt_r = make_reg(mesh, dx, mref_r, DEV)
R_m, Rt_m = make_reg(mesh, dx, mref_m, DEV)
mref_j = np.concatenate([mref_r, mref_m])
mref_jt = torch.tensor(mref_j, dtype=torch.float64, device=DEV)
m_j = torch.tensor(mref_j.copy(), dtype=torch.float64, device=DEV,
                   requires_grad=True)
LOB, HIB = float(np.log(1e-2)), float(np.log(1e6))
MLO = torch.tensor(np.r_[LOB * np.ones(nC), -2.0 * np.ones(nC)], device=DEV)
MHI = torch.tensor(np.r_[HIB * np.ones(nC), 2.0 * np.ones(nC)], device=DEV)
SL_R, SL_M = slice(0, nC), slice(nC, 2 * nC)


def dw_j(mm):
    d_r, d_i = eng.dpred(torch.clamp(mm, MLO, MHI))
    return torch.cat((W_r * d_r, np.sqrt(LAM) * W_i * d_i))


t0 = time.time()
beta_r, beta_m = beta0_blocks(dw_j, mref_jt, Rt_r, R_r, Rt_m, R_m, nC,
                              alpha_eta=ALPHA_M)
t_b0 = time.time() - t0
print(f"[beta0] rho {beta_r:.3e} | m {beta_m:.3e} (lapiz por bloque, "
      f"{t_b0:.1f}s)", flush=True)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
COOL, RATE, SUB, MAXIT = 4.0, 2, 5, 20
opt = torch.optim.LBFGS([m_j], lr=1.0, max_iter=SUB, history_size=10,
                        line_search_fn="strong_wolfe")
hist = []
for outer in range(MAX_OUTER):
    tA = time.time()
    pr = [0.0] * 4

    def closure():
        if m_j.grad is not None:
            m_j.grad = None
        mc = torch.clamp(m_j, MLO, MHI)
        d_r, d_i = eng.dpred(mc)
        f0 = 0.5 * torch.sum((W_r * (d_r - O_r)) ** 2)
        f1 = 0.5 * torch.sum((W_i * (d_i - O_i)) ** 2)
        g0 = quad(Rt_r, mc[SL_R] - mref_jt[SL_R])
        g1 = quad(Rt_m, mc[SL_M] - mref_jt[SL_M])
        loss = f0 + LAM * f1 + beta_r * g0 + beta_m * g1
        loss.backward()
        pr[0], pr[1] = float(f0.detach()), float(f1.detach())
        pr[2], pr[3] = float(g0.detach()), float(g1.detach())
        return loss

    for _ in range(MAXIT // SUB):
        opt.step(closure)
        if 2 * pr[0] / nD <= 1.0 and 2 * pr[1] / nD <= 1.0:
            break
    c_r, c_i = 2 * pr[0] / nD, 2 * pr[1] / nD
    hist.append((pr[0], pr[1], pr[2], pr[3], beta_r, beta_m,
                 time.time() - tA))
    print(f"  cr {outer+1:4d} chi2 rho {c_r:8.3f} ip {c_i:8.3f} | phi_m "
          f"{pr[2]:10.3e} {pr[3]:10.3e} | beta {beta_r:9.3e} {beta_m:9.3e}"
          f" | {time.time()-tA:5.1f}", flush=True)
    if c_r <= 1.0 and c_i <= 1.0:
        print("  cr *** TARGET: ambos chi2 <= 1", flush=True)
        break
    if (outer + 1) % RATE == 0:
        if c_r > 1.0:
            beta_r /= COOL if c_r > 4.0 else np.sqrt(COOL)
        if c_i > 1.0:
            beta_m /= COOL if c_i > 4.0 else np.sqrt(COOL)
t_loop = time.time() - t0
vram = torch.cuda.max_memory_reserved() / 2 ** 30
m_fin = torch.clamp(m_j, MLO, MHI).detach()
rho0_cr = m_fin[:nC].cpu().numpy()          # log rho_DC
m_cr = m_fin[nC:].cpu().numpy()             # cargabilidad
chi_r, chi_i = 2 * hist[-1][0] / nD, 2 * hist[-1][1] / nD
print(f"\n[cr] loop {t_loop:.1f}s | {len(hist)} outers | chi2 rho {chi_r:.3f}"
      f" ip {chi_i:.3f} | m: min {m_cr.min():.3f} max {m_cr.max():.3f} | "
      f"VRAM {vram:.2f} GiB", flush=True)

# ---------------------- comparacion a TRES bandas -------------------------
eta_true, mr_true = D["eta_true"], D["m_rho_true"]
# terreno comun: log rho_0 (la resistividad DC polarizada)
lr0_true = mr_true - np.log(1.0 - eta_true)
core = eta_true > 0.5 * CHRG["eta"]


def score(lr0, chg, name, chi, t, nout):
    return dict(name=name, chg_core=float(chg[core].mean()),
                chg_max=float(chg.max()), chg_min=float(chg.min()),
                chg_corr=float(np.corrcoef(chg, eta_true)[0, 1]),
                chg_rms=float(np.sqrt(np.mean((chg - eta_true) ** 2))),
                rho0_corr=float(np.corrcoef(lr0, lr0_true)[0, 1]),
                rho0_rms=float(np.sqrt(np.mean((lr0 - lr0_true) ** 2))),
                chi2_dc=round(chi[0], 3), chi2_ip=round(chi[1], 3),
                t=round(t, 1), outers=nout)


rows = [score(rho0_cr, m_cr, "complejo", (chi_r, chi_i), t_loop, len(hist))]
prev = os.path.join(REPO, "results", "inv_dcip_synth_f64.npz")
if os.path.exists(prev):
    p = np.load(prev)
    # secuencial: su rho recuperado ES rho_0 (ajusto d_DC = V1)
    rows.append(score(p["m_rho_seq"], p["eta_seq"], "secuencial",
                      (float(p["chi_seq_b"][0]), float(p["chi_seq_b"][1])),
                      float(p["t_dc"]) + float(p["t_lin"]), -1))
    # dos estados: rho_0 = rho_inf / (1-eta)
    lr0_j = p["m_rho_joint"] - np.log(1.0 - np.clip(p["eta_joint"], 0, 0.95))
    rows.append(score(lr0_j, p["eta_joint"], "dos-estados",
                      (float(p["chi_joint"][0]), float(p["chi_joint"][1])),
                      float(p["t_joint"]), int(p["hist_joint"].shape[0])))

print(f"\n===== TRES FORMULACIONES, mismo dato (eta_true = {CHRG['eta']}) =====")
print(f"{'':13s} {'chg nucleo':>11s} {'chg max':>9s} {'chg min':>9s} "
      f"{'chg corr':>9s} {'rho0 corr':>10s} {'chi2_DC':>8s} {'chi2_IP':>8s} "
      f"{'s':>7s}")
for r in rows:
    print(f"{r['name']:13s} {r['chg_core']:11.4f} {r['chg_max']:9.4f} "
          f"{r['chg_min']:9.4f} {r['chg_corr']:9.4f} {r['rho0_corr']:10.4f} "
          f"{r['chi2_dc']:8.3f} {r['chi2_ip']:8.3f} {r['t']:7.1f}")

out = os.path.join(REPO, "results", f"inv_cr_synth{TAG}.npz")
np.savez_compressed(out, log_rho_dc=rho0_cr, m=m_cr, hist=np.array(hist),
                    lr0_true=lr0_true, eta_true=eta_true, core=core,
                    chi_r=chi_r, chi_i=chi_i, t=t_loop, t_build=t_build,
                    t_b0=t_b0, nC=nC, nD=nD, dx=dx, vram=vram,
                    hx=mesh.h[0], hz=mesh.h[1], origin=mesh.origin)
print("\nJSON: " + json.dumps(dict(exp="cr-synth", rows=rows,
                                  vram_gib=round(vram, 2),
                                  total_s=round(time.time() - t00, 1))))
print(f"saved {out}", flush=True)
