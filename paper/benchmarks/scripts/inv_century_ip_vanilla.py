"""
inv_century_ip_vanilla.py — inversion IP linealizada de Century 46800E con
SimPEG 0.25 stock (receta del tutorial Transform-2020 MODERNIZADA):
induced_polarization.Simulation2DNodal storeJ + Pardiso, sigma=exp(-m_dc)
con el MISMO fondo DC que el motor dctorch (inv_century2d_dctorch_f64,
diferencia declarada vs tutorial), WeightedLeastSquares(alpha_s=1/dx²),
ProjectedGNCG(lower=0, upper=1000), BetaEstimate_ByEig(1) +
BetaSchedule(2,1) + TargetMisfit(chifact=1), std del OBS. eta en mV/V.

Correr (CWD scripts/, env simpeg311): python -u inv_century_ip_vanilla.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import time

import numpy as np
import psutil

from simpeg import (data, data_misfit, directives, inverse_problem, inversion,
                    maps, optimization, regularization)
from simpeg.electromagnetics.static import induced_polarization as ip
from pymatsolver import Pardiso

from century_common import REPO, ALPHA_X
from century_ip_common import (COOL_IP, COOLING_RATE_IP, ETA0,
                               build_dc_survey_and_mesh, build_ip_survey,
                               load_background)

_, mesh, dx = build_dc_survey_and_mesh()
m_dc, chi_dc = load_background(mesh)
survey, dobs, std = build_ip_survey()
nD, nC = int(survey.nD), int(mesh.nC)
print(f"[setup] IP nD {nD} | mesh {mesh.shape_cells} = {nC:,} celdas | "
      f"dx {dx:g} m | fondo DC chi {chi_dc:.3f}", flush=True)

m0 = ETA0 * np.ones(nC)
sim = ip.Simulation2DNodal(
    mesh, survey=survey, sigma=np.exp(-m_dc),
    etaMap=maps.IdentityMap(mesh), storeJ=True, solver=Pardiso)
dat = data.Data(survey, dobs=dobs, standard_deviation=std)
dmis = data_misfit.L2DataMisfit(data=dat, simulation=sim)
reg = regularization.WeightedLeastSquares(
    mesh, alpha_s=1.0 / dx ** 2, alpha_x=ALPHA_X, alpha_y=ALPHA_X,
    reference_model=m0)
opt = optimization.ProjectedGNCG(maxIter=40, maxIterLS=20, cg_maxiter=20,
                                 cg_rtol=1e-3, cg_atol=0.0,
                                 lower=0.0, upper=1000.0)
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
dirs = [directives.BetaEstimate_ByEig(beta0_ratio=1.0, random_seed=0),
        directives.BetaSchedule(coolingFactor=COOL_IP,
                                coolingRate=COOLING_RATE_IP),
        directives.TargetMisfit(chifact=1.0), timer]
inv = inversion.BaseInversion(invProb, directiveList=dirs)

t0 = time.time()
eta_rec = inv.run(m0.copy())
t_inv = time.time() - t0
phid = float(dmis(eta_rec))
chi = phid / nD
rss = psutil.Process().memory_info().peak_wset / 2**30
out = os.path.join(REPO, "results", "inv_century_ip_vanilla.npz")
np.savez_compressed(out, eta=eta_rec, t=t_inv, n_iters=int(opt.iter),
                    phid=phid, chi=chi, nC=nC, nD=nD, dx=dx, chi_dc=chi_dc,
                    iter_log=np.array(timer.rows))
print(f"\n[century-ip-vanilla] {t_inv:.1f}s | {int(opt.iter)} iters | "
      f"chi {chi:.3f} | RAM {rss:.2f} GiB", flush=True)
print("JSON: " + json.dumps(dict(engine="century-ip-GN-pardiso",
                                 t=round(t_inv, 1), iters=int(opt.iter),
                                 chi=round(chi, 3), nC=nC, nD=nD,
                                 ram_gib=round(rss, 2))))
print(f"saved {out}", flush=True)
