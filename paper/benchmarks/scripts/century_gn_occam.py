"""The Century line under a Gauss-Newton step on exact sensitivities.

Section 3.5 compares optimizers on this line and ranks the quasi-Newton
choice first. It leaves out the method the conventional workflow actually
uses, because that method wants the sensitivity matrix and this engine was
built not to form one. The batched adjoint makes the matrix cheap to form
when it fits, so the omission is worth closing: this runs
`TorchGaussNewton(beta_search=True)` on the same data, mesh, uncertainties,
regularization and target as the rest of that comparison, and counts cost in
objective evaluations, as that table does, so the number is comparable
without being a property of the machine.

Reported: evaluations, iterations, seconds, final chi, peak device memory by
NVML (the torch allocator does not see the driver context or cuDSS's
workspaces), and the correlation with the adopted L-BFGS solution.

  python -u paper/benchmarks/scripts/century_gn_occam.py
"""
import contextlib
import io
import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

from simpeg import (data, data_misfit, directives, inverse_problem,       # noqa: E402
                    inversion, maps, regularization)
from simpeg.electromagnetics.static import resistivity as dc              # noqa: E402

from century_common import (ALPHA_X, BETA0_RATIO, build_mesh,             # noqa: E402
                            build_survey, starting_halfspace)
from dctorch import TorchDC2D, TorchGaussNewton                           # noqa: E402

PREC = os.environ.get("PREC", "f32")
DT = torch.float32 if PREC == "f32" else torch.float64
OUT = os.environ.get("OUT", os.path.join(
    os.path.abspath(os.path.join(REPO_ROOT, "..", "_gpubench")),
    "century_gn_occam.json"))


class NVML:
    def __init__(self, period=0.02):
        import pynvml
        pynvml.nvmlInit()
        self._p, self._h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        self.period = period

    def _used(self):
        return self._p.nvmlDeviceGetMemoryInfo(self._h).used / 2 ** 20

    def __enter__(self):
        self.base = self.peak = self._used()
        self._stop = False

        def loop():
            while not self._stop:
                self.peak = max(self.peak, self._used())
                time.sleep(self.period)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.sec = time.perf_counter() - self.t0
        self._stop = True
        self._t.join()
        self.delta = self.peak - self.base


def main():
    survey, dobs, std = build_survey()
    mesh, dx = build_mesh(survey)
    rho0 = starting_halfspace(survey, dobs)
    nC, nD = mesh.n_cells, survey.nD
    m0 = np.log(rho0) * np.ones(nC)
    print(f"Century 46800E: {nD} data, {nC:,} cells, dx {dx:.0f} m, "
          f"rho0 {rho0:.0f} ohm-m, {PREC}", flush=True)

    eng = TorchDC2D(mesh, survey, nky=12, device="cuda", dtype=DT,
                    quadrature="electrodes")

    # Price the run the way Section 3.5 prices the others: in calls on the
    # forward operator, counted from outside so the engine stays unaware of
    # being measured. A Gauss-Newton step is not one evaluation -- it is a
    # forward per line-search probe plus one Jacobian -- so the three are
    # counted apart rather than folded into a single number that would
    # flatter whichever method defines "evaluation" its own way.
    n = {"dpred": 0, "grad": 0, "jac": 0}

    def counted(name, f):
        def g(*a, **k):
            n[name] += 1
            return f(*a, **k)
        return g

    eng.dpred = counted("dpred", eng.dpred)
    eng.misfit_and_grad = counted("grad", eng.misfit_and_grad)
    eng.Jmatrix = counted("jac", eng.Jmatrix)
    eng.linearize = counted("jac", eng.linearize)
    dat = data.Data(survey, dobs=dobs, standard_deviation=std)
    sim = dc.simulation_2d.Simulation2DNodal(mesh, survey=survey,
                                             rhoMap=maps.ExpMap(mesh), nky=eng.nky)
    reg = regularization.WeightedLeastSquares(
        mesh, alpha_s=1.0 / dx ** 2, alpha_x=ALPHA_X, alpha_y=ALPHA_X,
        reference_model=m0)
    dmis = data_misfit.L2DataMisfit(data=dat, simulation=sim)
    opt = TorchGaussNewton(engine=eng, maxIter=40, beta_search=True,
                           lower=np.log(1.0), upper=np.log(1e5))
    prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    inv = inversion.BaseInversion(prob, directiveList=[
        directives.BetaEstimate_ByEig(beta0_ratio=BETA0_RATIO),
        directives.TargetMisfit(chifact=1.0)])

    torch.cuda.synchronize()
    with NVML() as s:
        with contextlib.redirect_stdout(io.StringIO()):
            m = inv.run(m0)
        torch.cuda.synchronize()
    chi = float(np.sqrt(float(prob.phi_d) / nD))
    evals = n["dpred"] + n["grad"]
    res = dict(engine="dctorch GN-Occam", prec=PREC, nD=nD, nC=nC,
               sec=s.sec, nvml_mib=s.delta, iters=int(opt.iter),
               evals=evals, chi=chi, beta=float(prob.beta),
               n_dpred=n["dpred"], n_grad=n["grad"], n_jac=n["jac"])
    print(f"  {s.sec:6.1f} s | {opt.iter} iterations | {evals} forward "
          f"evaluations ({n['dpred']} dpred + {n['grad']} with gradient) | "
          f"{n['jac']} Jacobians | chi {chi:.3f} | NVML {s.delta:.0f} MiB",
          flush=True)

    ref = os.path.join(os.path.dirname(OUT), "inv_century2d_dctorch_f32.npz")
    if os.path.exists(ref):
        z = np.load(ref, allow_pickle=True)
        k = "m" if "m" in z else list(z.keys())[0]
        res["corr_lbfgs"] = float(np.corrcoef(m, np.asarray(z[k], float).ravel())[0, 1])
        print(f"  correlation with the adopted L-BFGS model "
              f"{res['corr_lbfgs']:.4f}", flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({k: (v.item() if hasattr(v, "item") else v)
                   for k, v in res.items()}, f, indent=2)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
