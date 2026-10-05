"""
inv_dcip_synth.py — EL experimento del peldano 2: mismo dato sintetico
(eta verdadera 0.40), dos flujos de trabajo.

  SECUENCIAL (la practica estandar, lo unico que permite SimPEG stock):
      1) inversion DC sola de d_DC  -> m_rho_seq
      2) inversion IP LINEALIZADA (TorchIP2D) con ese fondo -> eta_seq
  CONJUNTO (esto): inversion no-lineal simultanea de (log rho, logit eta)
      con TorchDCIP2D; gradiente exacto de AMBOS campos por AD, eta acotada
      por sigmoide dentro del grafo.

Diagnostico clave: los DOS modelos se evaluan con la MISMA fisica exacta
(el forward no-lineal) -> chi^2 honesto de cada flujo, ademas del error
contra la verdad.

Env: PREC=f64|f32 | LAM (peso del misfit IP, def 1) | ALPHA_ETA (def 1)
     MAXOUTER (def 40) | BETA0 (pencil|ratio, def pencil) | STAGE
Correr (CWD scripts/, env simpeg311): python -u inv_dcip_synth.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from dcip_joint import beta0_blocks, make_reg, quad, run_joint, t_sparse
from dcip_synth_common import (CHRG, REPO, build_survey_ip, make_data,
                               starting_halfspace)
from dctorch import TorchDC2D, TorchDCIP2D, TorchIP2D

PREC = os.environ.get("PREC", "f64")
DT = {"f64": torch.float64, "f32": torch.float32}[PREC]
LAM = float(os.environ.get("LAM", "1.0"))
ALPHA_ETA = float(os.environ.get("ALPHA_ETA", "1.0"))
MAX_OUTER = int(os.environ.get("MAXOUTER", "40"))
BETA0_KIND = os.environ.get("BETA0", "pencil")
DEV = "cuda"
# ETA0 = arranque/referencia. Con la sigmoide, d(eta)/d(m_eta) = eta(1-eta):
# arrancar en 0.01 amortigua la sensibilidad del bloque eta 100x y su beta0
# queda estimado en un regimen que deja de valer en cuanto eta crece.
ETA0 = float(os.environ.get("ETA0", "0.05"))
# cota superior FISICA de la cargabilidad (declarada, como las de log-rho):
# sin ella la sigmoide corre a 1 en las celdas sin sensibilidad.
ETA_MAX = float(os.environ.get("ETA_MAX", "0.6"))
WARM = os.environ.get("WARM", "0") == "1"     # arrancar del modelo secuencial
REG_SPACE = os.environ.get("REG_SPACE", "eta")   # eta (fisico) | logit
# CONTROL: que es el dato DC de la VERDAD sintetica. "total" = el voltaje
# polarizado (fisico para dominio del tiempo); "instantaneous" = la
# convencion que el flujo linealizado asume => aisla el error de
# LINEALIZACION puro, sin el error de convencion encima.
DC_STATE = os.environ.get("DC_STATE", "total")
ALPHA_X = 1.0
torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------- utilidades
def run_lbfgs(m, closure_factory, target, beta0, cool, cool_rate, maxouter,
              sub=None, maxiter=20, project=None, label=""):
    """Bucle estandar de la casa: L-BFGS(strong Wolfe) por nivel de beta,
    enfriamiento fijo, parada al target. `sub` parte el nivel en sub-pasos
    con chequeo de target entre ellos (necesario en problemas lineales)."""
    beta = beta0
    step = sub or maxiter
    opt = torch.optim.LBFGS([m], lr=1.0, max_iter=step, history_size=10,
                            line_search_fn="strong_wolfe")
    hist = []
    for outer in range(maxouter):
        tA = time.time()
        closure, probe = closure_factory(lambda: beta)
        for _ in range(max(maxiter // step, 1)):
            opt.step(closure)
            if project is not None:
                with torch.no_grad():
                    project(m)
            if probe[0] <= target:
                break
        hist.append((probe[0], probe[1], beta, time.time() - tA))
        print(f"{label}{outer+1:4d} {probe[0]:12.4e} {probe[1]:12.4e} "
              f"{beta:11.3e} {time.time()-tA:6.1f}", flush=True)
        if probe[0] <= target:
            print(f"{label}*** TARGET: phi_d {probe[0]:.1f} <= {target:.0f}",
                  flush=True)
            break
        if (outer + 1) % cool_rate == 0:
            beta /= cool
    return hist


# ------------------------------------------------------------------- el dato
t00 = time.time()
D = make_data(device=DEV, dc_state=DC_STATE)
mesh, dx, survey, nD = D["mesh"], D["dx"], D["survey"], D["nD"]
nC = int(mesh.nC)
rho0 = starting_halfspace(survey, D["dobs_dc"])
print(f"[setup] nD {nD} (x2 datasets) | {nC:,} celdas dx {dx:g} | malla de "
      f"generacion {D['mesh_fine'].nC:,} | rho0 {rho0:.1f} | eta_true max "
      f"{D['eta_true'].max():.2f} | prec {PREC} | eta0 {ETA0} eta_max "
      f"{ETA_MAX} alpha_eta {ALPHA_ETA} lam {LAM} warm {int(WARM)} | dc_state {DC_STATE}",
      flush=True)

W_dc = torch.tensor(1.0 / D["std_dc"], dtype=torch.float64, device=DEV)
W_ip = torch.tensor(1.0 / D["std_ip"], dtype=torch.float64, device=DEV)
O_dc = torch.tensor(D["dobs_dc"], dtype=torch.float64, device=DEV)
O_ip = torch.tensor(D["dobs_ip"], dtype=torch.float64, device=DEV)
target1 = 0.5 * nD                                # convencion ½, chifact 1
LOB, HIB = float(np.log(1e-2)), float(np.log(1e6))

# =========================================================== 1. DC secuencial
print("\n=== SECUENCIAL 1/2: inversion DC sola ===", flush=True)
t0 = time.time()
eng_dc = TorchDC2D(mesh, survey, device=DEV, dtype=DT)
mref_r = np.log(rho0) * np.ones(nC)
R_r, Rt_r = make_reg(mesh, dx, mref_r, DEV)
mref_rt = torch.tensor(mref_r, dtype=torch.float64, device=DEV)
t_build_dc = time.time() - t0

m_r = torch.tensor(mref_r.copy(), dtype=torch.float64, device=DEV,
                   requires_grad=True)


def _mk_dc(get_beta):
    probe = [None, None]

    def closure():
        opt_zero(m_r)
        mc = torch.clamp(m_r, LOB, HIB)
        fd = 0.5 * torch.sum(
            (W_dc * (eng_dc.dpred(mc).to(torch.float64) - O_dc)) ** 2)
        fm = quad(Rt_r, mc - mref_rt)
        loss = fd + get_beta() * fm
        loss.backward()
        probe[0], probe[1] = float(fd), float(fm)
        return loss
    return closure, probe


def opt_zero(t):
    if t.grad is not None:
        t.grad = None


with torch.no_grad():
    phid0 = 0.5 * float(torch.sum((W_dc * (eng_dc.dpred(mref_rt).to(
        torch.float64) - O_dc)) ** 2))
    pm0 = float(quad(Rt_r, 0.1 * torch.randn(nC, dtype=torch.float64,
                                             device=DEV)))
beta_dc = phid0 / (pm0 + 1e-30)
print(f"[dc] phi_d0 {phid0:.3e} | beta0 {beta_dc:.3e} | target {target1:.0f}",
      flush=True)
t0 = time.time()
h_dc = run_lbfgs(m_r, _mk_dc, target1, beta_dc, 4.0, 2, MAX_OUTER,
                 label="  dc ")
t_dc = time.time() - t0
m_rho_seq = torch.clamp(m_r, LOB, HIB).detach().cpu().numpy()
chi_dc_seq = 2 * h_dc[-1][0] / nD
eng_dc.solver.free()
del eng_dc
torch.cuda.empty_cache()
print(f"[dc] {t_dc:.1f}s | {len(h_dc)} outers | chi2 {chi_dc_seq:.3f}",
      flush=True)

# ================================================= 2. IP linealizado (Seigel)
print("\n=== SECUENCIAL 2/2: IP linealizado sobre ese fondo ===", flush=True)
t0 = time.time()
survey_ip = build_survey_ip()
eng_lin = TorchIP2D(mesh, survey_ip, m_rho_seq, device=DEV, dtype=DT)
t_build_lin = time.time() - t0
mref_e = ETA0 * np.ones(nC)
R_e, Rt_e = make_reg(mesh, dx, mref_e, DEV)
mref_et = torch.tensor(mref_e, dtype=torch.float64, device=DEV)
m_e = torch.tensor(mref_e.copy(), dtype=torch.float64, device=DEV,
                   requires_grad=True)
ELO, EHI = 0.0, 0.99


def _mk_lin(get_beta):
    probe = [None, None]

    def closure():
        opt_zero(m_e)
        mc = torch.clamp(m_e, ELO, EHI)
        fd = 0.5 * torch.sum(
            (W_ip * (eng_lin.dpred(mc).to(torch.float64) - O_ip)) ** 2)
        fm = quad(Rt_e, mc - mref_et)
        loss = fd + get_beta() * fm
        loss.backward()
        probe[0], probe[1] = float(fd), float(fm)
        return loss
    return closure, probe


with torch.no_grad():                 # operador LINEAL => J x = dpred(x)
    x = torch.randn(nC, dtype=torch.float64, device=DEV)
    num = float(torch.sum((W_ip * eng_lin.dpred(x).to(torch.float64)) ** 2))
    den = float(torch.sum(x.unsqueeze(1)
                          * torch.sparse.mm(Rt_e, x.unsqueeze(1))))
beta_lin = num / den
print(f"[lin] beta0 {beta_lin:.3e} (eig-ratio exacto)", flush=True)
t0 = time.time()
h_lin = run_lbfgs(m_e, _mk_lin, target1, beta_lin, 2.0, 1, MAX_OUTER, sub=5,
                  project=lambda t: t.clamp_(ELO, EHI), label="  ip ")
t_lin = time.time() - t0
eta_seq = torch.clamp(m_e, ELO, EHI).detach().cpu().numpy()
# El flujo secuencial NO define un par (sigma, eta) consistente: su modelo DC
# se ajusto al dato DC (que es el POLARIZADO) mientras el IP linealizado lo usa
# como si fuera la sigma instantanea. Las DOS lecturas posibles:
#   (a) "instantanea": m_rho_seq ES sigma        -> la que asume el IP lineal
#   (b) "total":       m_rho_seq es sigma(1-eta) -> la que asume el dato DC
m_rho_seq_inst = m_rho_seq + np.log(1.0 - np.clip(eta_seq, 0, 0.95))
chi_ip_seq_lin = 2 * h_lin[-1][0] / nD
eng_lin.dc.solver.free()
del eng_lin
torch.cuda.empty_cache()
print(f"[lin] {t_lin:.1f}s | {len(h_lin)} outers | chi2(lineal) "
      f"{chi_ip_seq_lin:.3f} | eta max {eta_seq.max():.4f}", flush=True)

# ============================================================== 3. CONJUNTO
print("\n=== CONJUNTO no-lineal (rho, eta) ===", flush=True)
t0 = time.time()
eng = TorchDCIP2D(mesh, survey, device=DEV, dtype=DT, dc_state=DC_STATE,
                  eta_max=ETA_MAX)
t_build_j = time.time() - t0
mref_me = eng.m_eta_of(ETA0 * np.ones(nC))
mref_j = np.concatenate([mref_r, mref_me])
mref_jt = torch.tensor(mref_j, dtype=torch.float64, device=DEV)
if WARM:      # arranque tibio = el modelo secuencial, releido consistente
    m0_j = np.concatenate([m_rho_seq_inst,
                           eng.m_eta_of(np.clip(eta_seq, 1e-3,
                                                0.95 * ETA_MAX))])
else:
    m0_j = mref_j.copy()
m_j = torch.tensor(m0_j, dtype=torch.float64, device=DEV,
                   requires_grad=True)
SL_R, SL_E = slice(0, nC), slice(nC, 2 * nC)
MLO = torch.tensor(np.r_[LOB * np.ones(nC), -12.0 * np.ones(nC)], device=DEV)
MHI = torch.tensor(np.r_[HIB * np.ones(nC), 12.0 * np.ones(nC)], device=DEV)


def dw_j(mm):
    """dato PESADO y apilado (el que define J para el lapiz)."""
    d_dc, d_ip = eng.dpred(torch.clamp(mm, MLO, MHI))
    return torch.cat((W_dc * d_dc.to(torch.float64),
                      np.sqrt(LAM) * W_ip * d_ip.to(torch.float64)))


# beta0 POR CAMPO: el lapiz restringido a cada bloque. Un solo beta obliga a
# los dos campos a compartir schedule y el bloque rho acaba sin regularizar
# mientras eta pide libertad (medido: rho corr 0.53 vs 0.82 del DC solo).
t0 = time.time()
if BETA0_KIND == "pencil":
    beta_r_j, beta_e_j = beta0_blocks(dw_j, mref_jt, Rt_r, R_r, Rt_e, R_e,
                                      nC, alpha_eta=ALPHA_ETA)
    if REG_SPACE == "eta":
        # el lapiz mide R en el espacio del PARAMETRO (logit); al regularizar
        # en el FISICO, d(eta) = s*d(m_eta) en la referencia => la forma
        # cuadratica escala por s^2 y beta0 por 1/s^2.
        _sg = ETA0 / ETA_MAX
        beta_e_j /= (ETA_MAX * _sg * (1.0 - _sg)) ** 2
    print(f"[joint] beta0 rho {beta_r_j:.3e} | eta {beta_e_j:.3e} "
          f"(lapiz por bloque, {time.time()-t0:.1f}s)", flush=True)
else:
    with torch.no_grad():
        d0, i0 = eng.dpred(mref_jt)
        pd0 = 0.5 * float(torch.sum((W_dc * (d0 - O_dc)) ** 2))
        pi0 = 0.5 * float(torch.sum((W_ip * (i0 - O_ip)) ** 2))
        pm0j = float(quad(Rt_r, 0.1 * torch.randn(nC, dtype=torch.float64,
                                                  device=DEV)))
    beta_r_j, beta_e_j = pd0 / pm0j, ALPHA_ETA * pi0 / pm0j
    print(f"[joint] beta0 rho {beta_r_j:.3e} | eta {beta_e_j:.3e} (ratio)",
          flush=True)
t_b0 = time.time() - t0

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
eta_ref_t = (torch.full((nC,), ETA0, dtype=torch.float64, device=DEV)
             if REG_SPACE == "eta" else None)
h_j, m_fin, _ = run_joint(eng, m_j, mref_jt, Rt_r, Rt_e, W_dc, O_dc, W_ip,
                          O_ip, nC, nD, nD, beta_r_j, beta_e_j, (MLO, MHI),
                          lam=LAM, maxouter=MAX_OUTER, eta_ref=eta_ref_t)
t_joint = time.time() - t0
vram = torch.cuda.max_memory_reserved() / 2 ** 30
m_rho_joint = m_fin[:nC].cpu().numpy()
eta_joint = eng.eta_of(m_fin[nC:]).cpu().numpy()
print(f"[joint] {t_joint:.1f}s | {len(h_j)} outers | eta max "
      f"{eta_joint.max():.4f} | VRAM {vram:.2f} GiB", flush=True)


# ==================================================== 4. diagnostico honesto
def chi2_exact(m_rho, eta):
    """chi^2 de CADA dataset (+ los datos predichos) evaluando el modelo con
    la fisica EXACTA: el unico juez comun a los dos flujos."""
    with torch.no_grad():
        d_dc, d_ip = eng.dpred_from_fields(
            torch.tensor(m_rho, dtype=torch.float64, device=DEV),
            torch.tensor(np.clip(eta, 1e-12, 0.999 * ETA_MAX),
                         dtype=torch.float64, device=DEV))
        c0 = float(torch.mean((W_dc * (d_dc - O_dc)) ** 2))
        c1 = float(torch.mean((W_ip * (d_ip - O_ip)) ** 2))
    return (c0, c1), d_dc.cpu().numpy(), d_ip.cpu().numpy()


chi_seq_a, pred_a_dc, pred_a_ip = chi2_exact(m_rho_seq, eta_seq)
chi_seq_b, pred_b_dc, pred_b_ip = chi2_exact(m_rho_seq_inst, eta_seq)
# la lectura VALIDA del modelo secuencial depende de que sea el dato DC
pred_seq_dc, pred_seq_ip = ((pred_a_dc, pred_a_ip)
                            if DC_STATE == "instantaneous"
                            else (pred_b_dc, pred_b_ip))
chi_seq_used = chi_seq_a if DC_STATE == "instantaneous" else chi_seq_b
chi_joint, pred_joint_dc, pred_joint_ip = chi2_exact(m_rho_joint, eta_joint)
eta_true, mr_true = D["eta_true"], D["m_rho_true"]
core = eta_true > 0.5 * CHRG["eta"]              # el cuerpo cargable


def score(m_rho, eta):
    return dict(
        eta_corr=float(np.corrcoef(eta, eta_true)[0, 1]),
        eta_rms=float(np.sqrt(np.mean((eta - eta_true) ** 2))),
        eta_core_mean=float(eta[core].mean()),
        eta_core_max=float(eta[core].max()),
        eta_max=float(eta.max()),
        rho_corr=float(np.corrcoef(m_rho, mr_true)[0, 1]),
        rho_rms=float(np.sqrt(np.mean((m_rho - mr_true) ** 2))))


s_seq = score(m_rho_seq, eta_seq)
s_seq_b = score(m_rho_seq_inst, eta_seq)
s_joint = score(m_rho_joint, eta_joint)
print("\n=== RESULTADO ===", flush=True)
print(f"eta verdadera en el cuerpo: {CHRG['eta']:.3f} "
      f"({int(core.sum())} celdas) | sesgo analitico del linealizado "
      f"eta/(1+eta) = {CHRG['eta']/(1+CHRG['eta']):.3f}", flush=True)
for name, s, c, t in (("secuencial (lectura instantanea)", s_seq, chi_seq_a,
                       t_dc + t_lin),
                      ("secuencial (lectura total)      ", s_seq_b, chi_seq_b,
                       t_dc + t_lin),
                      ("conjunto no-lineal              ", s_joint, chi_joint,
                       t_joint)):
    print(f"[{name}] eta nucleo media {s['eta_core_mean']:.4f} max "
          f"{s['eta_core_max']:.4f} | corr {s['eta_corr']:.4f} rms "
          f"{s['eta_rms']:.4f} | rho corr {s['rho_corr']:.4f} | "
          f"chi2 exacto DC {c[0]:.3f} IP {c[1]:.3f} | {t:.1f}s", flush=True)

TAG = os.environ.get("TAG", "")
out = os.path.join(REPO, "results", f"inv_dcip_synth_{PREC}{TAG}.npz")
np.savez_compressed(
    out, m_rho_true=mr_true, eta_true=eta_true, m_rho_seq=m_rho_seq,
    eta_seq=eta_seq, m_rho_joint=m_rho_joint, eta_joint=eta_joint,
    hist_dc=np.array(h_dc), hist_lin=np.array(h_lin), hist_joint=np.array(h_j),
    dobs_dc=D["dobs_dc"], dobs_ip=D["dobs_ip"], std_dc=D["std_dc"],
    std_ip=D["std_ip"], clean_ip=D["clean_ip"], nC=nC, nD=nD, dx=dx,
    hx=mesh.h[0], hz=mesh.h[1], origin=mesh.origin, core=core,
    t_dc=t_dc, t_lin=t_lin, t_joint=t_joint, t_b0=t_b0, vram=vram,
    m_rho_seq_inst=m_rho_seq_inst, pred_seq_dc=pred_seq_dc,
    pred_seq_ip=pred_seq_ip, pred_joint_dc=pred_joint_dc,
    pred_joint_ip=pred_joint_ip, chi_seq_used=np.array(chi_seq_used),
    chi_seq_a=np.array(chi_seq_a),
    chi_seq_b=np.array(chi_seq_b), chi_joint=np.array(chi_joint), prec=PREC,
    lam=LAM, alpha_eta=ALPHA_ETA, dc_state=DC_STATE)
print("JSON: " + json.dumps(dict(
    exp="dcip-synth", prec=PREC, lam=LAM, alpha_eta=ALPHA_ETA,
    eta0=ETA0, eta_max=ETA_MAX, warm=WARM, tag=TAG, dc_state=DC_STATE,
    reg_space=REG_SPACE,
    seq={**{k: round(v, 4) for k, v in s_seq.items()},
         "chi2_dc": round(chi_seq_a[0], 3), "chi2_ip": round(chi_seq_a[1], 3),
         "rho_corr_total": round(s_seq_b["rho_corr"], 4),
         "chi2_dc_total": round(chi_seq_b[0], 3),
         "chi2_ip_total": round(chi_seq_b[1], 3),
         "t": round(t_dc + t_lin, 1), "outers": len(h_dc) + len(h_lin)},
    joint={**{k: round(v, 4) for k, v in s_joint.items()},
           "chi2_dc": round(chi_joint[0], 3), "chi2_ip": round(chi_joint[1], 3),
           "t": round(t_joint, 1), "outers": len(h_j)},
    vram_gib=round(vram, 2), total_s=round(time.time() - t00, 1))))
print(f"saved {out}", flush=True)
