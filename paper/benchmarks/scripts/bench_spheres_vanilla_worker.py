"""
bench_spheres_vanilla_worker.py — UN punto vanilla (SimPEG 0.25 + Pardiso
CPU) del escalado reproducible dos-esferas: t_fwd (fields+dpred con
factorizacion fresca) y t_grad (Jtvec(WtWr) reusando fields), storeJ=False.
Env: DH, NREP (def 2). Emite "JSON: {...}".
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from simpeg import maps
from simpeg.electromagnetics.static import resistivity as dc
from pymatsolver import Pardiso

from spheres_common import build_problem, starting_logrho

DH = float(os.environ["DH"])
NREP = int(os.environ.get("NREP", "2"))
np.random.seed(0)

mesh, survey, dobs, std, active, nC = build_problem(DH)
print(f"[mesh] dh={DH:g} | active {nC:,} | nodes {mesh.n_nodes:,} | "
      f"nD {survey.nD}", flush=True)

rmap = maps.InjectActiveCells(mesh, active_cells=active,
                              value_inactive=1e8) * maps.ExpMap(nP=nC)
sim = dc.simulation.Simulation3DNodal(mesh, survey=survey, rhoMap=rmap,
                                      storeJ=False, solver=Pardiso)
W = 1.0 / std
m = starting_logrho(survey, dobs) * np.ones(nC)

t0 = time.time()
f = sim.fields(m)
d = sim.dpred(m, f=f)
t_cold = time.time() - t0
print(f"[cold] {t_cold:.1f}s", flush=True)

fwd, grad = [], []
for _ in range(NREP):
    m2 = m + np.random.randn(nC) * 1e-3
    t0 = time.time()
    f = sim.fields(m2)
    d = sim.dpred(m2, f=f)
    fwd.append(time.time() - t0)
    v = W * (W * (d - dobs))
    t0 = time.time()
    g = sim.Jtvec(m2, v, f=f)
    grad.append(time.time() - t0)
    print(f"  fwd {fwd[-1]:.1f}s | Jtvec {grad[-1]:.1f}s", flush=True)

rss = psutil.Process().memory_info().peak_wset / 2**30
print("JSON: " + json.dumps(dict(
    dh=DH, engine="vanilla-pardiso", problem="spheres", n_active=nC,
    n_nodes=int(mesh.n_nodes), nD=int(survey.nD), t_cold=round(t_cold, 1),
    t_fwd=round(float(np.median(fwd)), 2),
    t_grad=round(float(np.median(grad)), 2),
    ram_peak_gib=round(rss, 2))), flush=True)
