"""
inv_century_cr.py — Century 46800E con RESISTIVIDAD COMPLEJA (la formulacion
de RES2DINV / cR2 / CRTomo: sigma* = sigma_DC(1 - i*m)), sobre el dato PUBLICO
real: 46800POT.OBS + 46800IP.OBS invertidos simultaneamente.

Tercera de las tres formulaciones sobre el mismo dato de campo:
  1. secuencial linealizado  (inv_century2d_vanilla + inv_century_ip_vanilla,
     codigo STOCK de SimPEG de punta a punta)
  2. conjunta dos estados    (inv_century_dcip.py)
  3. compleja                (este)

Emparejamiento de datos (gotcha medido en el sintetico): el dato DC de Century
es NEGATIVO en todos los cuadripolos; el observable de la formulacion compleja
es la MAGNITUD |V*| (de ahi sale la resistividad aparente) => se compara
|V*| contra |V_obs|. La fase Im/Re es invariante al signo.

Env: MAXOUTER (def 40) | ALPHA_M (def 1) | LAM (def 1) | M0 (def 0.005) | TAG
Correr (CWD scripts/, env simpeg311): python -u inv_century_cr.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from century_common import starting_halfspace
from century_dcip_common import IP_SCALE, REPO, build
from dcip_joint import beta0_blocks, make_reg, quad
from dctorch import TorchCR2D

MAX_OUTER = int(os.environ.get("MAXOUTER", "40"))
ALPHA_M = float(os.environ.get("ALPHA_M", "1.0"))
LAM = float(os.environ.get("LAM", "1.0"))
M0 = float(os.environ.get("M0", "0.005"))      # 5 mV/V ~ mediana del dato
# Enfriamiento ACOPLADO (default) = el esquema de RES2DINV: el amortiguamiento
# del IP es un RATIO FIJO del de resistividad, nunca independiente. Con
# enfriamiento independiente en Century los dos betas se separan 12 DECADAS
# (el IP llega a target en el outer 13 y congela beta_m en 1.3e3 mientras
# beta_rho cae a 1e-9) y el L-BFGS unico se estanca: chi2_DC clavado en 16.3
# con 40 outers agotados, aunque el DC solo se invierte a chi2 0.965.
COOL_MODE = os.environ.get("COOL_MODE", "coupled")   # coupled | independent
TAG = os.environ.get("TAG", "")
DEV = "cuda"
torch.manual_seed(0)
np.random.seed(0)

t00 = time.time()
survey, obs_dc, std_dc, obs_ip, std_ip, mesh, dx = build()
nD, nC = int(survey.nD), int(mesh.nC)
rho0 = starting_halfspace(survey, obs_dc)
neg = int((obs_dc < 0).sum())
print(f"[setup] nD {nD} x2 | {nC:,} celdas ({2*nC:,} parametros) dx {dx:g} | "
      f"rho0 {rho0:.1f} | POT negativos {neg}/{nD} | IP "
      f"{obs_ip.min():.1f}-{obs_ip.max():.1f} mV/V", flush=True)

t0 = time.time()
eng = TorchCR2D(mesh, survey, device=DEV, dtype=torch.complex128,
                ip_scale=IP_SCALE, dc_datum="magnitude")
t_build = time.time() - t0
print(f"[build] {t_build:.1f}s | solver COMPLEJO SIMETRICO", flush=True)

W_r = torch.tensor(1.0 / std_dc, dtype=torch.float64, device=DEV)
W_i = torch.tensor(1.0 / std_ip, dtype=torch.float64, device=DEV)
O_r = torch.tensor(np.abs(obs_dc), dtype=torch.float64, device=DEV)
O_i = torch.tensor(obs_ip, dtype=torch.float64, device=DEV)

mref_r = np.log(rho0) * np.ones(nC)
mref_m = M0 * np.ones(nC)                      # SIN transformacion, SIN cotas
R_r, Rt_r = make_reg(mesh, dx, mref_r, DEV)
R_m, Rt_m = make_reg(mesh, dx, mref_m, DEV)
mref_j = np.concatenate([mref_r, mref_m])
mref_jt = torch.tensor(mref_j, dtype=torch.float64, device=DEV)
m_j = torch.tensor(mref_j.copy(), dtype=torch.float64, device=DEV,
                   requires_grad=True)
LOB, HIB = float(np.log(1e-2)), float(np.log(1e6))
MLO = torch.tensor(np.r_[LOB * np.ones(nC), -1.0 * np.ones(nC)], device=DEV)
MHI = torch.tensor(np.r_[HIB * np.ones(nC), 1.0 * np.ones(nC)], device=DEV)
SL_R, SL_M = slice(0, nC), slice(nC, 2 * nC)


S = 1.0            # provisional: el lapiz corre en la parametrizacion fisica


def dw_j(mm):
    mc = torch.clamp(mm, MLO, MHI)
    d_r, d_i = eng.dpred_from_fields(mc[SL_R], S * mc[SL_M])
    return torch.cat((W_r * d_r, np.sqrt(LAM) * W_i * d_i))


t0 = time.time()
beta_r, beta_m = beta0_blocks(dw_j, mref_jt, Rt_r, R_r, Rt_m, R_m, nC,
                              alpha_eta=ALPHA_M)
# PRECONDICIONADOR del bloque de cargabilidad. m cruda es O(5e-3) contra
# log(rho) ~ 4.9, y con ip_scale=1000 la curvatura de los dos bloques difiere
# en ordenes: el L-BFGS UNICO se estanca (chi2_DC clavado en 16-22 con 40
# outers, aunque el DC solo llega a 0.965 y el forward es identico al DC real
# a 1.5e-15 en el arranque). El lapiz ya da la escala: beta ~ curvatura, y
# reescalar m por s multiplica su curvatura por s^2 => s = sqrt(beta_r/beta_m)
# iguala los dos bloques. Es el analogo del logit del peldano 2: la libertad
# de no tener cotas se paga con tener que escalar a mano.
S = float(np.sqrt(beta_r / beta_m))
beta_m = beta_m * S ** 2                       # == beta_r por construccion
mref_m = mref_m / S
mref_j = np.concatenate([mref_r, mref_m])
mref_jt = torch.tensor(mref_j, dtype=torch.float64, device=DEV)
m_j = torch.tensor(mref_j.copy(), dtype=torch.float64, device=DEV,
                   requires_grad=True)
MLO = torch.tensor(np.r_[LOB * np.ones(nC), -1.0 / S * np.ones(nC)],
                   device=DEV)
MHI = torch.tensor(np.r_[HIB * np.ones(nC), 1.0 / S * np.ones(nC)],
                   device=DEV)
print(f"[precond] s = {S:.4e}  (m_fisica = s * m_param; m_param de "
      f"referencia = {mref_m[0]:.3f})", flush=True)
t_b0 = time.time() - t0
print(f"[beta0] rho {beta_r:.3e} | m {beta_m:.3e} (ratio {beta_m/beta_r:.3e},"
      f" lapiz por bloque, {t_b0:.1f}s) | cooling {COOL_MODE}", flush=True)

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
        d_r, d_i = eng.dpred_from_fields(mc[SL_R], S * mc[SL_M])
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
        if COOL_MODE == "coupled":       # ratio fijo, estilo RES2DINV
            cw = max(c_r, c_i)
            f = COOL if cw > 4.0 else np.sqrt(COOL)
            beta_r, beta_m = beta_r / f, beta_m / f
        else:
            if c_r > 1.0:
                beta_r /= COOL if c_r > 4.0 else np.sqrt(COOL)
            if c_i > 1.0:
                beta_m /= COOL if c_i > 4.0 else np.sqrt(COOL)

t_loop = time.time() - t0
vram = torch.cuda.max_memory_reserved() / 2 ** 30
m_fin = torch.clamp(m_j, MLO, MHI).detach()
log_rho_dc = m_fin[:nC].cpu().numpy()
m_cr = S * m_fin[nC:].cpu().numpy()            # de vuelta a la m FISICA
chi_r, chi_i = 2 * hist[-1][0] / nD, 2 * hist[-1][1] / nD
print(f"\n[century-cr] loop {t_loop:.1f}s (build {t_build:.1f} + beta0 "
      f"{t_b0:.1f}) | {len(hist)} outers | chi2 rho {chi_r:.3f} ip "
      f"{chi_i:.3f} | m: min {m_cr.min()*IP_SCALE:.1f} max "
      f"{m_cr.max()*IP_SCALE:.1f} mV/V | VRAM {vram:.2f} GiB", flush=True)

out = os.path.join(REPO, "results", f"inv_century_cr_f64{TAG}.npz")
np.savez_compressed(out, log_rho_dc=log_rho_dc, m=m_cr, hist=np.array(hist),
                    chi_r=chi_r, chi_i=chi_i, t=t_loop, t_build=t_build,
                    t_b0=t_b0, nC=nC, nD=nD, dx=dx, rho0=rho0, vram=vram,
                    m0=M0, alpha_m=ALPHA_M, lam=LAM)
print("JSON: " + json.dumps(dict(
    engine="century-cr-f64", t=round(t_loop, 1), t_build=round(t_build, 1),
    t_b0=round(t_b0, 1), iters=len(hist), chi2_dc=round(chi_r, 3),
    chi2_ip=round(chi_i, 3), nC=nC, nD=nD,
    m_max_mVV=round(float(m_cr.max()) * IP_SCALE, 2),
    m_min_mVV=round(float(m_cr.min()) * IP_SCALE, 2),
    m_med_mVV=round(float(np.median(m_cr)) * IP_SCALE, 3),
    vram_gib=round(vram, 2), total_s=round(time.time() - t00, 1))))
print(f"saved {out}", flush=True)
