"""
inv_tutorial3d_dctorch.py — el MISMO problema del tutorial oficial (dos
esferas, dc_data.xyz descargado) invertido con dctorch: misma malla, mismas
incertidumbres (1e-7+10%|d|), misma WeightedLeastSquares ls=100, mismo
schedule (cooling /2 cada 2) y mismo target (chifact=1: phi_d(sum)<=nD, en
nuestra convencion ½: pd<=nD/2). Modelo en log-rho = -log-sigma (identico
fisicamente; la regularizacion de gradientes es invariante al signo).

Diferencias declaradas vs el tutorial (nuestro flujo estandar): L-BFGS(20)
por nivel de beta con persist entre niveles NO (script simple), beta0 =
100*phid0/pm0 (vs BetaEstimate_ByEig(100)), sin sensitivity weights, clamp
anti-overflow MUY ancho [0.1, 1e7] ohm-m (inactivo en la practica: el modelo
verdadero vive en 10-1000).

SENSW=1 (default): sensitivity weights como el tutorial (UpdateSensitivity
Weights every_iteration, threshold 1e-2), estimados SIN formar J via
Hutchinson: E[(Jt W v)^2]_j = diag(Jt W^2 J)_j con v Rademacher, K=SENSW_K
(64) backwards que reusan la factorizacion; la R se rearma por outer con
reg.set_weights(sensitivity=w) (la MISMA mecanica de SimPEG). SENSW=0 = loop
anterior sin pesos.

Env: DH (def 60), DCTORCH_PREC=f32|f64, MAXOUTER (def 40 como el tutorial).
Salida: inv_tutorial3d_dctorch_{prec}_dh{DH}.npz + JSON.
Correr bajo watchdog (CWD _gpubench, env simpeg311).
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

from simpeg.electromagnetics.static import resistivity as dc  # noqa: F401
from simpeg.electromagnetics.static.utils.static_utils import (
    apparent_resistivity_from_voltage)
from simpeg.utils.io_utils.io_utils_electromagnetics import read_dcip_xyz
from simpeg.utils import model_builder
from simpeg import maps, regularization
from discretize import TreeMesh
from discretize.utils import active_from_xyz

from dctorch import TorchDC3D

HERE = (os.environ.get("DCTORCH_BENCH_OUT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "_gpubench")))
DH = float(os.environ.get("DH", "60"))
PREC = os.environ.get("DCTORCH_PREC", "f32")
DT = {"f64": torch.float64, "f32": torch.float32}[PREC]
MAX_OUTER = int(os.environ.get("MAXOUTER", "40"))
SENSW = os.environ.get("SENSW", "1") == "1"
SENSW_K = int(os.environ.get("SENSW_K", "64"))
SENSW_THRESH = 1e-2
LBFGS_MAXITER = 20
COOL, COOLING_RATE, BETA0_RATIO = 2.0, 2, 100.0
torch.manual_seed(0)
np.random.seed(0)

# ---------- datos del tutorial (mismo cache que el vanilla) ----------
dir_path = os.path.join(HERE, "inv_dcr_3d_files") + os.path.sep
assert os.path.exists(os.path.join(dir_path, "topo_xyz.txt")), \
    "correr inv_tutorial3d_vanilla.py primero (descarga)"
topo_xyz = np.loadtxt(dir_path + "topo_xyz.txt")
dc_data, _ = read_dcip_xyz(
    dir_path + "dc_data.xyz", "volt", data_header="V/A",
    uncertainties_header="UNCERT", is_surface_data=False,
    dict_headers=["LINEID"])
std = 1e-7 + 0.1 * np.abs(dc_data.dobs)
dobs = dc_data.dobs

# ---------- malla (calcada del tutorial) ----------
nbc = lambda w: 2 ** int(np.round(np.log(w / DH) / np.log(2.0)))  # noqa: E731
mesh = TreeMesh([[(DH, nbc(8000.))], [(DH, nbc(8000.))], [(DH, nbc(4000.))]],
                x0="CCN", diagonal_balance=True)
mesh.origin = mesh.origin + np.r_[0.0, 0.0, topo_xyz[:, -1].max()]
k = np.sqrt(np.sum(topo_xyz[:, 0:2] ** 2, axis=1)) < 1200
mesh.refine_surface(topo_xyz[k, :], padding_cells_by_level=[0, 4, 4],
                    finalize=False)
mesh.refine_points(dc_data.survey.unique_electrode_locations,
                   padding_cells_by_level=[6, 6, 4], finalize=False)
mesh.finalize()
active = active_from_xyz(mesh, topo_xyz)
nC = int(active.sum())
dc_data.survey.drape_electrodes_on_topography(mesh, active,
                                              topo_cell_cutoff="top")
nD = int(dc_data.survey.nD)
print(f"[mesh] dh={DH:g} | active {nC:,} | nodes {mesh.n_nodes:,} | nD {nD}",
      flush=True)

# ---------- motor (log-rho; val_inactive 1e8 ohm-m == 1e-8 S/m del tutorial) --
t0 = time.time()
eng = TorchDC3D(mesh, dc_data.survey, device="cuda", dtype=DT,
                active_cells=active, val_inactive=np.log(1e8))
t_build = time.time() - t0
print(f"[build] {t_build:.1f}s | precision {PREC}", flush=True)

# ---------- modelo inicial + regularizacion (equivalentes al tutorial) ------
app_cond = 1 / apparent_resistivity_from_voltage(dc_data.survey, dobs)
rho0 = float(1.0 / np.median(app_cond))
mref = np.log(rho0) * np.ones(nC)          # = -log(median_conductivity)
reg = regularization.WeightedLeastSquares(
    mesh, active_cells=active, length_scale_x=100.0, length_scale_y=100.0,
    length_scale_z=100.0, reference_model=mref)


def build_R():
    Rc = sp.csr_matrix(reg.deriv2(mref)).tocoo()
    return torch.sparse_coo_tensor(
        torch.tensor(np.vstack((Rc.row, Rc.col)), dtype=torch.int64,
                     device="cuda"),
        torch.tensor(Rc.data, dtype=torch.float64, device="cuda"),
        Rc.shape).coalesce()


R_t = build_R()
mref_t = torch.tensor(mref, dtype=torch.float64, device="cuda")
W_t = torch.tensor(1.0 / std, dtype=torch.float64, device="cuda")
dobs_t = torch.tensor(dobs, dtype=torch.float64, device="cuda")
LOB, HIB = float(np.log(0.1)), float(np.log(1e7))   # anti-overflow, inactivo
clamp = lambda m: torch.clamp(m, LOB, HIB)          # noqa: E731


def phi_m(mc):
    d2 = (mc - mref_t).unsqueeze(1)
    return 0.5 * torch.sum(d2 * torch.sparse.mm(R_t, d2))


def phi_d(mc):
    d = eng.dpred(mc).to(torch.float64)
    return 0.5 * torch.sum((W_t * (d - dobs_t)) ** 2)


def update_sens_weights(m_leaf):
    """w = sqrt(diag(Jt W^2 J)) via Hutchinson (K backwards, reusa el factor),
    normalizado a max=1 y con piso SENSW_THRESH — calca UpdateSensitivity
    Weights del tutorial; rearma R con reg.set_weights (mecanica SimPEG)."""
    global R_t
    mc = clamp(m_leaf)
    d = eng.dpred(mc).to(torch.float64)
    acc = torch.zeros(nC, dtype=torch.float64, device="cuda")
    for k in range(SENSW_K):
        v = torch.where(torch.rand(len(dobs), device="cuda") < 0.5,
                        -1.0, 1.0).to(torch.float64)
        (g,) = torch.autograd.grad(d, m_leaf, grad_outputs=W_t * v,
                                   retain_graph=(k < SENSW_K - 1))
        acc += g ** 2
    w = torch.sqrt(acc / SENSW_K)
    w = torch.clamp(w / w.max(), min=SENSW_THRESH)
    reg.set_weights(sensitivity=w.cpu().numpy())
    R_t = build_R()


target = 0.5 * nD                # == chifact 1 en la convencion sin ½ de 0.25
m = torch.tensor(mref.copy(), dtype=torch.float64, device="cuda",
                 requires_grad=True)
t_sensw = 0.0
if SENSW:
    t0 = time.time()
    update_sens_weights(m)
    t_sensw += time.time() - t0
    print(f"[sensw] K={SENSW_K} Hutchinson | {t_sensw:.1f}s el primero",
          flush=True)
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
    if SENSW and outer > 0:                  # every_iteration, como el tutorial
        update_sens_weights(m)

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
chi = 2 * hist[-1][0] / nD                    # convencion sin ½, como tutorial
m_np = clamp(m).detach().cpu().numpy()

# score vs modelo verdadero (mismo del tutorial, en log; signo de rho)
true_sigma = 1e-2 * np.ones(nC)
cc = mesh.cell_centers[active, :]
true_sigma[model_builder.get_indices_sphere(
    np.r_[-300.0, 0.0, 100.0], 165.0, cc)] = 1e-1
true_sigma[model_builder.get_indices_sphere(
    np.r_[300.0, 0.0, 100.0], 165.0, cc)] = 1e-3
corr_true = float(np.corrcoef(np.log(true_sigma), -m_np)[0, 1])

print(f"\n[dctorch-{PREC} dh={DH:g}] loop {t_loop:.1f}s (build {t_build:.1f}s)"
      f" | {len(hist)} outers | chi {chi:.3f} | corr vs verdad {corr_true:.4f}"
      f" | VRAM {vram:.2f} GiB", flush=True)
_tag = "" if SENSW else "_nosw"
out = os.path.join(HERE, f"inv_tutorial3d_dctorch_{PREC}_dh{DH:g}{_tag}.npz")
np.savez_compressed(out, m_log_rho=m_np, hist=np.array(hist), t=t_loop,
                    t_build=t_build, chi=chi, corr_true=corr_true, nC=nC,
                    dh=DH, prec=PREC, sensw=SENSW)
print("JSON: " + json.dumps(dict(dh=DH, engine=f"dctorch-{PREC}",
                                 t=round(t_loop, 1), t_build=round(t_build, 1),
                                 iters=len(hist), chi=round(chi, 3),
                                 corr_true=round(corr_true, 4), nC=nC,
                                 vram_gib=round(vram, 2), sensw=SENSW)))
print(f"saved {out}", flush=True)
