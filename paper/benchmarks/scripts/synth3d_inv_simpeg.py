"""synth3d_inv_simpeg.py — invert the field-scale synthetic with stock SimPEG 0.25.

Same data, mesh, electrode placement, starting/reference model, regularisation
(WeightedLeastSquares, 100 m length scales), beta0 rule and stopping level as
synth3d_inv_dctorch.py. Model is log-conductivity (the SimPEG convention), i.e.
minus this engine's log-resistivity; the quadratic regularisation does not see
the sign. Solver: SimPEG's default (Pardiso), default threading.

MODE
  gn_storej  the official tutorial recipe: InexactGaussNewton(maxIter=40,
             maxIterLS=20, cg_maxiter=30, cg_rtol=1e-3) with the sensitivities
             stored and UpdatePreconditioner (Jacobi from diag JtJ), beta
             halved every 2 iterations. Fastest stock path while J fits in RAM.
  gn         the same Gauss-Newton with storeJ=False: J.v and Jt.v by adjoint
             solves that reuse the factorisation, sources looped one by one in
             _Jtvec. No preconditioner: getJtJdiag builds the full J even when
             storeJ=False, which would defeat the point of this mode.
  bfgs       SimPEG's limited-memory BFGS, storeJ=False: the optimiser family
             this engine uses, on the CPU. Beta halved every 40 iterations to
             match the engine's 2 outer iterations of 20 L-BFGS steps.
No sensitivity weights in any mode (they also call getJ).

Convention: SimPEG's phi_d is sum r^2 (no 1/2), so chi2 = phi_d / nD; beta0 =
100 phi_d0 / phi_m(dm) in that convention is twice the engine's, which gives
the same minimiser.

Stopping: chi2 at the accepted model <= chi2 of the true model on this mesh,
or maxIter, or TIME_CAP_S counted from the end of the mesh build. The cap is
hard: a watchdog stops the run at the cap (mid-iteration if needed) and records
the last completed iteration.

Env: MODE (gn_storej), DH (30), TIME_CAP_S (21600), NO_RESULT (0), MAXITER.
Launch: cmd /c "cd /d <scripts> && call activate simpeg311 && set MODE=gn&& set DH=30&& python -u synth3d_inv_simpeg.py"
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import threading
import time

import numpy as np

from simpeg import (maps, data_misfit, regularization, optimization,
                    inverse_problem, inversion, directives)
from simpeg import data as sdata
from simpeg.electromagnetics.static import resistivity as dc

import synth3d_common as S
import synth3d_bench as Bn

MODE = os.environ.get("MODE", "gn_storej")
DH = float(os.environ.get("DH", "30"))
TIME_CAP = float(os.environ.get("TIME_CAP_S", "21600"))
NO_RESULT = os.environ.get("NO_RESULT", "0") == "1"
assert MODE in ("gn_storej", "gn", "bfgs")
MAXITER = int(os.environ.get("MAXITER", "40" if MODE != "bfgs" else "800"))
LENGTH_SCALE, BETA0_RATIO = 100.0, 100.0

sampler = Bn.PeakSampler(gpu=False).start()
survey = S.build_survey()
t0 = time.time()
mesh, active = S.build_mesh(DH, survey, S.topo_points())
nC = int(active.sum())
t_mesh = time.time() - t0
d_obs, std = Bn.load_obs()
assert len(d_obs) == survey.nD, "data do not match the survey"
nD = len(d_obs)
target_chi2 = float(os.environ["TARGET_CHI2"]) if os.environ.get("TARGET_CHI2") else Bn.chi2_target(DH)
RUN_TAG = os.environ.get("RUN_TAG", "")
print(f"[mesh] dh={DH:g} | active {nC:,} | nodes {mesh.n_nodes:,} | nD {nD} | mode {MODE} | "
      f"target chi2 {target_chi2} | {t_mesh:.0f}s", flush=True)

t_work = time.time()
data = sdata.Data(survey, dobs=d_obs, standard_deviation=std)
sigma_map = maps.InjectActiveCells(mesh, active, 1e-8) * maps.ExpMap(nP=nC)
sim = dc.simulation.Simulation3DNodal(mesh, survey=survey, sigmaMap=sigma_map,
                                      storeJ=(MODE == "gn_storej"))
rho0 = Bn.starting_rho(survey, d_obs)
m0 = -np.log(rho0) * np.ones(nC)                       # log-conductivity
dmis = data_misfit.L2DataMisfit(simulation=sim, data=data)
reg = regularization.WeightedLeastSquares(
    mesh, active_cells=active, length_scale_x=LENGTH_SCALE,
    length_scale_y=LENGTH_SCALE, length_scale_z=LENGTH_SCALE, reference_model=m0)
phid0 = float(dmis(m0))
beta0 = BETA0_RATIO * phid0 / float(reg(m0 - Bn.perturbation(nC)))
print(f"[init] rho0 {rho0:.1f} | chi2_0 {phid0/nD:.2f} | beta0 {beta0:.4e} | "
      f"{time.time()-t_work:.0f}s (includes the first forward)", flush=True)

if MODE == "bfgs":
    opt = optimization.BFGS(maxIter=MAXITER, maxIterLS=20)
    cooling = directives.BetaSchedule(coolingFactor=2.0, coolingRate=40)
else:
    opt = optimization.InexactGaussNewton(maxIter=MAXITER, maxIterLS=20,
                                          cg_maxiter=30, cg_rtol=1e-3)
    cooling = directives.BetaSchedule(coolingFactor=2.0, coolingRate=2)
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=beta0)

state = dict(hist=[], stop="maxiter", reached=None, t_loop0=None, done=False)


class Monitor(directives.InversionDirective):
    """Per-iteration log; stops at the truth level of chi2 or at the time cap."""

    def initialize(self):
        state["t_loop0"] = time.time()

    def endIter(self):
        chi2 = float(self.invProb.phi_d) / nD
        t = time.time() - state["t_loop0"]
        state["hist"].append((int(self.opt.iter), t, chi2, float(self.invProb.phi_m),
                              float(self.invProb.beta)))
        print(f"  it {self.opt.iter:4d} | chi2 {chi2:9.3f} | beta {self.invProb.beta:.3e} | "
              f"{t:8.1f}s | host peak {sampler.peak_rss:.2f} GiB", flush=True)
        state["m"] = self.invProb.model.copy()
        if target_chi2 is not None and chi2 <= target_chi2:
            state["stop"], state["reached"] = "target", int(self.opt.iter)
            self.opt.stopNextIteration = True
        elif time.time() - t_work > TIME_CAP:
            state["stop"] = "time_cap"
            self.opt.stopNextIteration = True


dirs = [cooling, Monitor()]
if MODE == "gn_storej":
    dirs.insert(0, directives.UpdatePreconditioner(update_every_iteration=True))
inv = inversion.BaseInversion(inv_prob, dirs)


def finish(m_log_sigma, partial=False):
    if state["done"]:
        return
    state["done"] = True
    mem = sampler.stop()
    hist = state["hist"]
    scores = Bn.truth_metrics(mesh, active, -m_log_sigma, DH) if m_log_sigma is not None else {}
    row = dict(engine="simpeg", tag=RUN_TAG, solver="Pardiso-CPU", mode=MODE,
               optimizer="BFGS" if MODE == "bfgs" else "InexactGN",
               store_j=MODE == "gn_storej", dh=DH, nC=nC, nN=int(mesh.n_nodes), nD=nD,
               target_chi2=target_chi2,
               stop=(state.get("abort", "watchdog") if partial else state["stop"]),
               reached_at_iter=state["reached"], iters=len(hist),
               chi2_final=round(hist[-1][2], 4) if hist else None,
               t_mesh_s=round(t_mesh, 1),
               t_loop_s=round(time.time() - state["t_loop0"], 1) if state["t_loop0"] else None,
               t_total_after_mesh_s=round(time.time() - t_work, 1), **mem, **scores,
               hardware=Bn.hardware(gpu=False))
    print("RESULT: " + json.dumps(row), flush=True)
    if not NO_RESULT:
        Bn.write_result(row)
        np.savez_compressed(os.path.join(Bn.OUT, f"bench_simpeg_{MODE}_dh{DH:g}{RUN_TAG}.npz"),
                            m_log_rho=(-m_log_sigma if m_log_sigma is not None else np.array([])),
                            hist=np.array(hist), row=json.dumps(row))


RAM_TOTAL = __import__("psutil").virtual_memory().total / 2 ** 30


def watchdog():
    """Time cap overrun, or host memory above 90 % of RAM (Windows would page
    for hours instead of failing): write what exists and exit."""
    while not state["done"]:
        time.sleep(2)
        if sampler.peak_rss > 0.9 * RAM_TOTAL:
            state["abort"] = "memory_guard"
            print(f"[watchdog] host memory {sampler.peak_rss:.1f} GiB > 90 % of {RAM_TOTAL:.1f} GiB", flush=True)
            finish(state.get("m"), partial=True)
            os._exit(0)
        if time.time() - t_work > TIME_CAP:
            state["abort"] = "time_cap"
            print(f"[watchdog] {TIME_CAP:.0f} s after the mesh: stopping; partial result from "
                  f"the last completed iteration", flush=True)
            finish(state.get("m"), partial=True)
            os._exit(0)


threading.Thread(target=watchdog, daemon=True).start()
m_final = inv.run(m0)
finish(m_final)
