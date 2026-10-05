"""
ip_parity_century.py — GATE de paridad del IP linealizado (peldano 1):
TorchIP2D (dctorch, cuDSS GPU) vs induced_polarization.Simulation2DNodal
(SimPEG 0.25 stock, Pardiso CPU) sobre la linea Century 46800E con el fondo
DC recuperado (inv_century2d_dctorch_f64.npz). Sin paridad NO se avanza.

Checks:
  1. dpred(eta) en 3 modelos eta (uniforme 1e-3 / bump gaussiano 20 mV/V /
     uniforme aleatorio [0,10] mV/V) — objetivo ~1e-14 relativo
  2. gradiente: J^T v por autograd (dctorch) vs sim.Jtvec (stock), v aleatorio
  3. FD direccional sobre TorchIP2D (operador lineal => exacto)

Correr (CWD scripts/, env simpeg311): python -u ip_parity_century.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from simpeg import maps
from simpeg.electromagnetics.static import induced_polarization as ip
from pymatsolver import Pardiso

from century_ip_common import (build_dc_survey_and_mesh, build_ip_survey,
                               load_background)
from dctorch import TorchIP2D

NKY = 11

survey_dc, mesh, dx = build_dc_survey_and_mesh()
m_dc, chi_dc = load_background(mesh)
survey_ip, dobs_ip, std_ip = build_ip_survey()
nC, nD = int(mesh.nC), int(survey_ip.nD)
print(f"[setup] malla {mesh.shape_cells} = {nC:,} celdas (dx {dx:g}) | "
      f"IP nD {nD} | fondo DC chi {chi_dc:.3f} | dobs_ip "
      f"[{dobs_ip.min():.1f}, {dobs_ip.max():.1f}] mV/V", flush=True)

# ---- stock reference -------------------------------------------------------
t0 = time.time()
sim = ip.Simulation2DNodal(
    mesh, survey=survey_ip, sigma=np.exp(-m_dc),
    etaMap=maps.IdentityMap(mesh), solver=Pardiso, nky=NKY)
rng = np.random.default_rng(11)
cc_ = mesh.cell_centers
bump = 20.0 * np.exp(-(((cc_[:, 0] - 26900.0) / 300.0) ** 2
                       + ((cc_[:, 1] + 150.0) / 120.0) ** 2))
etas = {"uniform1e-3": 1e-3 * np.ones(nC),
        "bump20": bump,
        "rand[0,10]": rng.uniform(0.0, 10.0, nC)}
d_stock = {k: np.asarray(sim.dpred(e)) for k, e in etas.items()}
t_stock = time.time() - t0
print(f"[stock] sim + 3 dpred: {t_stock:.1f}s | "
      f"d(bump20) rango [{d_stock['bump20'].min():.3e}, "
      f"{d_stock['bump20'].max():.3e}]", flush=True)

# ---- dctorch ---------------------------------------------------------------
t0 = time.time()
eng = TorchIP2D(mesh, survey_ip, m_dc, nky=NKY, device="cuda",
                dtype=torch.float64)
print(f"[build] TorchIP2D: {time.time()-t0:.1f}s", flush=True)

ok = True
for k, e in etas.items():
    d_t = eng.dpred(torch.tensor(e, device="cuda")).cpu().numpy()
    ref = d_stock[k]
    rel2 = np.linalg.norm(d_t - ref) / np.linalg.norm(ref)
    reli = np.abs(d_t - ref).max() / np.abs(ref).max()
    flag = "OK " if rel2 < 1e-13 else "FAIL"
    ok &= rel2 < 1e-13
    print(f"[paridad dpred] {k:14s} rel2 {rel2:.2e} | rel-inf {reli:.2e} "
          f"[{flag}]", flush=True)

# ---- gradiente: J^T v (autograd) vs Jtvec (stock) --------------------------
eta0 = etas["bump20"]
v = rng.standard_normal(nD)
g_stock = np.asarray(sim.Jtvec(eta0, v))
et = torch.tensor(eta0, device="cuda", requires_grad=True)
vt = torch.tensor(v, device="cuda")
(eng.dpred(et) * vt).sum().backward()
g_t = et.grad.cpu().numpy()
relg = np.linalg.norm(g_t - g_stock) / np.linalg.norm(g_stock)
flag = "OK " if relg < 1e-13 else "FAIL"
ok &= relg < 1e-13
print(f"[paridad grad]  J^T v autograd vs stock Jtvec: rel2 {relg:.2e} "
      f"[{flag}]", flush=True)

# ---- FD direccional sobre el operador lineal -------------------------------
w = torch.tensor(1.0 / std_ip, device="cuda")
db = torch.tensor(dobs_ip, device="cuda")
p = torch.tensor(rng.standard_normal(nC), device="cuda")
et = torch.tensor(eta0, device="cuda", requires_grad=True)
phi = eng.misfit(et, db, w)
phi.backward()
gp = float((et.grad * p).sum())
errs = []
for eps in (1e-3, 1e-5):
    with torch.no_grad():
        fp = float(eng.misfit(et + eps * p, db, w))
        fm = float(eng.misfit(et - eps * p, db, w))
    fd = (fp - fm) / (2 * eps)
    errs.append(abs(fd - gp) / abs(fd))
    print(f"[FD] eps {eps:.0e}: FD {fd:.10e} vs g.p {gp:.10e} "
          f"rel {errs[-1]:.2e}", flush=True)
ok &= min(errs) < 1e-7

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                   "results", "ip_parity_century.npz")
np.savez_compressed(out, m_dc=m_dc, **{f"eta_{i}": e for i, e in
                                       enumerate(etas.values())},
                    **{f"dstock_{i}": d for i, d in
                       enumerate(d_stock.values())})
print(f"\n[GATE] {'PARIDAD OK' if ok else '*** FALLA DE PARIDAD ***'}",
      flush=True)
sys.exit(0 if ok else 1)
