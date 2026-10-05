"""
inv_century_dcip.py — inversion CONJUNTA no-lineal (rho, eta) de la linea
Century 46800E con dato PUBLICO real: 46800POT.OBS (volt) + 46800IP.OBS
(mV/V) invertidos SIMULTANEAMENTE, sin fijar el fondo y sin linealizar.

Contraste con el peldano 1 (`inv_century_ip_dctorch.py`): alli el fondo era
el modelo DC ya recuperado y eta salia de un problema LINEAL; aqui los dos
campos se estiman a la vez, la eta esta acotada por sigmoide dentro del grafo
y el dato DC se modela como el voltaje POLARIZADO (dc_state="total"), que es
lo que mide un receptor de dominio del tiempo.

Century es un caso de eta PEQUENA (dato 0.8-17.6 mV/V => eta ~ 1-2%), asi que
se espera que conjunto y secuencial CONVERJAN: eso es justamente el gate de
cordura sobre dato real (el sintetico cubre el regimen de eta grande).

Env: PREC=f64|f32 | ALPHA_ETA (def 1) | LAM (def 1) | MAXOUTER (def 40)
     ETA0 (def 0.005 V/V) | ETA_MAX (def 0.3) | TAG
Correr (CWD scripts/, env simpeg311): python -u inv_century_dcip.py
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
from dcip_joint import beta0_blocks, make_reg, run_joint
from dctorch import TorchDCIP2D

PREC = os.environ.get("PREC", "f64")
DT = {"f64": torch.float64, "f32": torch.float32}[PREC]
LAM = float(os.environ.get("LAM", "1.0"))
ALPHA_ETA = float(os.environ.get("ALPHA_ETA", "1.0"))
MAX_OUTER = int(os.environ.get("MAXOUTER", "40"))
ETA0 = float(os.environ.get("ETA0", "0.005"))     # 5 mV/V ~ mediana del dato
ETA_MAX = float(os.environ.get("ETA_MAX", "0.3"))
REG_SPACE = os.environ.get("REG_SPACE", "eta")    # eta | logit
TAG = os.environ.get("TAG", "")
DEV = "cuda"
torch.manual_seed(0)
np.random.seed(0)

t00 = time.time()
survey, obs_dc, std_dc, obs_ip, std_ip, mesh, dx = build()
nD, nC = int(survey.nD), int(mesh.nC)
rho0 = starting_halfspace(survey, obs_dc)
print(f"[setup] nD {nD} x2 | {nC:,} celdas dx {dx:g} | rho0 {rho0:.1f} | "
      f"IP {obs_ip.min():.1f}-{obs_ip.max():.1f} mV/V (mediana "
      f"{np.median(obs_ip):.1f}) | eta0 {ETA0} eta_max {ETA_MAX} | {PREC}",
      flush=True)

t0 = time.time()
eng = TorchDCIP2D(mesh, survey, device=DEV, dtype=DT, dc_state="total",
                  eta_max=ETA_MAX, ip_scale=IP_SCALE)
t_build = time.time() - t0
print(f"[build] {t_build:.1f}s", flush=True)

W_dc = torch.tensor(1.0 / std_dc, dtype=torch.float64, device=DEV)
W_ip = torch.tensor(1.0 / std_ip, dtype=torch.float64, device=DEV)
O_dc = torch.tensor(obs_dc, dtype=torch.float64, device=DEV)
O_ip = torch.tensor(obs_ip, dtype=torch.float64, device=DEV)

mref_r = np.log(rho0) * np.ones(nC)
mref_e = eng.m_eta_of(ETA0 * np.ones(nC))
R_r, Rt_r = make_reg(mesh, dx, mref_r, DEV)
R_e, Rt_e = make_reg(mesh, dx, mref_e, DEV)
mref_j = np.concatenate([mref_r, mref_e])
mref_jt = torch.tensor(mref_j, dtype=torch.float64, device=DEV)
m_j = torch.tensor(mref_j.copy(), dtype=torch.float64, device=DEV,
                   requires_grad=True)
LOB, HIB = float(np.log(1e-2)), float(np.log(1e6))
MLO = torch.tensor(np.r_[LOB * np.ones(nC), -12.0 * np.ones(nC)], device=DEV)
MHI = torch.tensor(np.r_[HIB * np.ones(nC), 12.0 * np.ones(nC)], device=DEV)


def dw_j(mm):
    d_dc, d_ip = eng.dpred(torch.clamp(mm, MLO, MHI))
    return torch.cat((W_dc * d_dc.to(torch.float64),
                      np.sqrt(LAM) * W_ip * d_ip.to(torch.float64)))


t0 = time.time()
beta_r, beta_e = beta0_blocks(dw_j, mref_jt, Rt_r, R_r, Rt_e, R_e, nC,
                              alpha_eta=ALPHA_ETA)
eta_ref = None
if REG_SPACE == "eta":
    # el lapiz mide R en el espacio del PARAMETRO (logit); al regularizar en
    # el fisico, d(eta) = s*d(m_eta) con s = d(eta)/d(m_eta) en la referencia
    # => la forma cuadratica se escala por s^2 y beta0 por 1/s^2.
    sig0 = ETA0 / ETA_MAX
    s = ETA_MAX * sig0 * (1.0 - sig0)
    beta_e /= s ** 2
    eta_ref = torch.full((nC,), ETA0, dtype=torch.float64, device=DEV)
    print(f"[reg] eta regularizada en el espacio FISICO (s={s:.3e})",
          flush=True)
t_b0 = time.time() - t0
print(f"[beta0] rho {beta_r:.3e} | eta {beta_e:.3e} (lapiz por bloque, "
      f"{t_b0:.1f}s)", flush=True)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
hist, m_fin, betas = run_joint(eng, m_j, mref_jt, Rt_r, Rt_e, W_dc, O_dc,
                               W_ip, O_ip, nC, nD, nD, beta_r, beta_e,
                               (MLO, MHI), lam=LAM, maxouter=MAX_OUTER,
                               eta_ref=eta_ref)
t_loop = time.time() - t0
vram = torch.cuda.max_memory_reserved() / 2 ** 30
m_rho = m_fin[:nC].cpu().numpy()
eta = eng.eta_of(m_fin[nC:]).cpu().numpy()
chi_dc, chi_ip = 2 * hist[-1][0] / nD, 2 * hist[-1][1] / nD

# comparacion con el peldano 1 (IP linealizado sobre el fondo DC ya invertido)
cmp_txt = ""
p1 = os.path.join(REPO, "results", "inv_century_ip_dctorch_f64.npz")
if os.path.exists(p1):
    d1 = np.load(p1)
    eta_lin = d1["eta"] / IP_SCALE                  # mV/V -> V/V
    if eta_lin.size == nC:
        cmp_txt = (f" | vs peldano 1 (lineal): corr "
                   f"{np.corrcoef(eta, eta_lin)[0, 1]:.4f}, max lineal "
                   f"{eta_lin.max()*IP_SCALE:.1f} mV/V")

print(f"\n[century-dcip-{PREC}] loop {t_loop:.1f}s (build {t_build:.1f} + "
      f"beta0 {t_b0:.1f}) | {len(hist)} outers | chi2 DC {chi_dc:.3f} IP "
      f"{chi_ip:.3f} | eta max {eta.max()*IP_SCALE:.1f} mV/V | VRAM "
      f"{vram:.2f} GiB{cmp_txt}", flush=True)

out = os.path.join(REPO, "results", f"inv_century_dcip_{PREC}{TAG}.npz")
np.savez_compressed(out, m_log_rho=m_rho, eta=eta, hist=np.array(hist),
                    t=t_loop, t_build=t_build, t_b0=t_b0, chi_dc=chi_dc,
                    chi_ip=chi_ip, nC=nC, nD=nD, dx=dx, rho0=rho0, prec=PREC,
                    lam=LAM, alpha_eta=ALPHA_ETA, eta0=ETA0, eta_max=ETA_MAX,
                    reg_space=REG_SPACE,
                    vram=vram)
print("JSON: " + json.dumps(dict(
    engine=f"century-dcip-{PREC}", t=round(t_loop, 1),
    t_build=round(t_build, 1), t_b0=round(t_b0, 1), iters=len(hist),
    chi2_dc=round(chi_dc, 3), chi2_ip=round(chi_ip, 3), nC=nC, nD=nD,
    eta_max_mVV=round(float(eta.max()) * IP_SCALE, 2),
    eta_med_mVV=round(float(np.median(eta)) * IP_SCALE, 3),
    vram_gib=round(vram, 2), total_s=round(time.time() - t00, 1))))
print(f"saved {out}", flush=True)
