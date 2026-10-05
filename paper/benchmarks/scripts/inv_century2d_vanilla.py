"""
inv_century2d_vanilla.py — Century 2D (linea 46800E, dato publico SimPEG)
con el flujo del tutorial Transform-2020 MODERNIZADO a simpeg 0.25:
Simulation2DNodal storeJ + Pardiso, WeightedLeastSquares(alpha_s=1/dx²,
alpha_x=alpha_y=1), InexactGaussNewton(20, cg_maxiter=20),
BetaEstimate_ByEig(1) + BetaSchedule(4,2) + TargetMisfit(chifact=1),
std del archivo OBS. IterTimer por iteracion (endIter).

Salida: results de paper/benchmarks + JSON en stdout.
Correr (CWD _gpubench, env simpeg311): python -u inv_century2d_vanilla.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import time

import numpy as np
import psutil

from simpeg import (data, data_misfit, directives, inverse_problem, inversion,
                    maps, optimization, regularization)
from simpeg.electromagnetics.static import resistivity as dc  # noqa: F401
from pymatsolver import Pardiso

from century_common import (REPO, BETA0_RATIO, COOL, COOLING_RATE, ALPHA_X,
                            build_mesh, build_survey, starting_halfspace)

survey, dobs, std = build_survey()
mesh, dx = build_mesh(survey)
nD = int(survey.nD)
nC = int(mesh.nC)
print(f"[setup] nD {nD} | mesh {mesh.shape_cells} = {nC:,} celdas | "
      f"dx {dx:g} m", flush=True)

rho0 = starting_halfspace(survey, dobs)
m0 = np.log(rho0) * np.ones(nC)
print(f"[start] rho0 {rho0:.1f} ohm-m", flush=True)

sim = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, rhoMap=maps.ExpMap(mesh), storeJ=True,
    solver=Pardiso)
dat = data.Data(survey, dobs=dobs, standard_deviation=std)
dmis = data_misfit.L2DataMisfit(data=dat, simulation=sim)
reg = regularization.WeightedLeastSquares(
    mesh, alpha_s=1.0 / dx ** 2, alpha_x=ALPHA_X, alpha_y=ALPHA_X,
    reference_model=m0)
opt = optimization.InexactGaussNewton(maxIter=20, maxIterLS=20,
                                      cg_maxiter=20, cg_rtol=1e-3)
invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)


class IterTimer(directives.InversionDirective):
    def initialize(self):
        self.rows = []
        self.t0 = time.time()

    def endIter(self):
        self.rows.append((self.opt.iter, time.time() - self.t0,
                          float(self.invProb.phi_d),
                          float(self.invProb.phi_m),
                          float(self.invProb.beta)))


timer = IterTimer()
dirs = [directives.BetaEstimate_ByEig(beta0_ratio=BETA0_RATIO, random_seed=0),
        directives.BetaSchedule(coolingFactor=COOL, coolingRate=COOLING_RATE),
        directives.TargetMisfit(chifact=1.0), timer]
inv = inversion.BaseInversion(invProb, directiveList=dirs)

t0 = time.time()
m_rec = inv.run(m0.copy())
t_inv = time.time() - t0
phid = float(dmis(m_rec))
chi = phid / nD
rss = psutil.Process().memory_info().peak_wset / 2**30
out = os.path.join(REPO, "results", "inv_century2d_vanilla.npz")
np.savez_compressed(out, m_log_rho=m_rec, t=t_inv, n_iters=int(opt.iter),
                    phid=phid, chi=chi, nC=nC, nD=nD, dx=dx, rho0=rho0,
                    iter_log=np.array(timer.rows))
print(f"\n[century-vanilla] {t_inv:.1f}s | {int(opt.iter)} iters | "
      f"chi {chi:.3f} | RAM {rss:.2f} GiB", flush=True)
print("JSON: " + json.dumps(dict(engine="century-GN-pardiso",
                                 t=round(t_inv, 1), iters=int(opt.iter),
                                 chi=round(chi, 3), nC=nC, nD=nD,
                                 ram_gib=round(rss, 2))))
print(f"saved {out}", flush=True)
