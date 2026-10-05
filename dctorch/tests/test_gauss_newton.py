"""TorchGaussNewton: the pieces of a step, then whole inversions.

  (a) the assembled regularization Hessian H_m v equals the autograd
      Hessian-vector product of the extracted phi_m;
  (b) the gradient a step takes from J, J^T (W dphi/dr), equals the autograd
      gradient of phi_d -- for the L2 misfit, Student-t, and the log-data
      misfit (ln|d|, J scaled by 1/d);
  (c) the data-space direct solve (Woodbury) and CG run to convergence give
      the same Gauss-Newton step;
  (d) a whole inversion reaches the target misfit, and the explicit and
      matrix-free paths land on the same model (to the round-off a CG
      truncated at 1e-2 lets through: the products agree to 1e-15, the
      truncation point can amplify that to ~1e-6);
  (e) TorchLBFGS, now sharing the base class, still inverts the same data;
  (f) the Occam beta search: its step at the beta it picks IS the exact
      Gauss-Newton step there, its runs end near the target, and two runs
      whose beta0 differ by 10^4 end on nearly the same beta.

  python -m dctorch.tests.test_gauss_newton
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import contextlib
import io

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64


def rel(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def problem(seed=0, noise=0.03):
    from simpeg import data, maps
    from simpeg.electromagnetics.static import resistivity as dc
    from dctorch import TorchDC2D
    from dctorch.tests.test_suite import make_case

    mesh, survey, _ = make_case("2d-tensor")
    cc = mesh.cell_centers
    m_true = np.log(100.0) * np.ones(mesh.nC)
    m_true[(np.abs(cc[:, 0]) < 6) & (cc[:, 1] < -3) & (cc[:, 1] > -9)] = np.log(10.0)
    eng = TorchDC2D(mesh, survey, nky=11, device=DEV)
    with torch.no_grad():
        dclean = eng.dpred(torch.tensor(m_true, dtype=DT, device=DEV)).cpu().numpy()
    rng = np.random.default_rng(seed)
    std = noise * np.abs(dclean)
    dat = data.Data(survey, dobs=dclean + std * rng.standard_normal(dclean.size),
                    standard_deviation=std)
    sim = dc.simulation_2d.Simulation2DNodal(mesh, survey=survey,
                                             rhoMap=maps.ExpMap(mesh), nky=11)
    return mesh, eng, dat, sim


def inv_problem(mesh, eng, dat, sim, opt, misfit="l2", beta=1.0):
    from simpeg import data_misfit, inverse_problem, regularization
    from dctorch import StudentTDataMisfit

    m0 = np.log(100.0) * np.ones(mesh.nC)
    reg = regularization.WeightedLeastSquares(mesh, alpha_s=0.16, alpha_x=1.0,
                                              alpha_y=1.0, reference_model=m0)
    if misfit == "l2":
        dmis = data_misfit.L2DataMisfit(data=dat, simulation=sim)
    else:
        dmis = StudentTDataMisfit(nu=4.0, data=dat, simulation=sim)
    prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    prob.beta = beta
    return prob, m0


def pieces():
    from dctorch import TorchGaussNewton
    from dctorch.optimization import _pcg

    mesh, eng, dat, sim = problem()
    for misfit, logd in (("l2", False), ("student-t", False), ("l2", True)):
        opt = TorchGaussNewton(engine=eng, jacobian="explicit", log_data=logd)
        prob, m0 = inv_problem(mesh, eng, dat, sim, opt, misfit=misfit, beta=0.5)
        x = m0 + 0.2 * np.random.default_rng(1).standard_normal(m0.size)
        prob.phi_d = prob.phi_m = np.nan      # what an inversion run initializes
        opt.startup(x)
        opt._torch_setup()
        phi_m_t = opt._build_phi_m_t()
        mt = torch.tensor(x, dtype=DT, device=DEV)

        # (a) assembled H_m against autograd
        v = torch.randn(mesh.nC, dtype=DT, device=DEV)
        Hm, Hm_diag = opt._phi_m_hessian()
        _, hvp = torch.autograd.functional.hvp(phi_m_t, mt, v)
        e_h = rel(Hm(v), hvp)

        # (b) gradient through J against autograd of phi_d
        post = opt._eval_point(x)
        opt._fill_gradient(post)
        m_g = mt.clone().requires_grad_(True)
        fd, _ = opt._phi_d_t(m_g)
        g_ref, = torch.autograd.grad(fd, m_g)
        e_g = rel(torch.as_tensor(post["g_d"], device=DEV), g_ref)

        # (c) direct step vs CG to convergence, on the same system
        J = opt._sensitivities(post["m"])["J"]
        d = torch.as_tensor(post["dpred"], dtype=DT, device=DEV)
        _, D = opt._residual_weights(d)
        g = torch.as_tensor(post["g"], dtype=DT, device=DEV)
        beta = float(prob.beta)
        p_dir = opt._solve_direct(J, D, g, beta)
        diag = (D.unsqueeze(1) * J ** 2).sum(0) + beta * Hm_diag
        p_cg, it = _pcg(lambda u: J.T @ (D * (J @ u)) + beta * Hm(u), -g,
                        1.0 / diag, 5000, 1e-12)
        e_s = rel(p_dir, p_cg)
        # (f) the Occam step at the beta it chose equals the direct step there
        r = opt._residual(d)
        g_d = torch.as_tensor(post["g_d"], dtype=DT, device=DEV)
        g_m = torch.as_tensor(post["g_m"], dtype=DT, device=DEV)
        b_c, p_occ, _ = opt._occam(J, D, g_d, g_m, r, beta, int(D.numel()),
                                   w=opt._lin_weight(d))
        p_ref = opt._solve_direct(J, D, g_d + b_c * g_m, b_c)
        e_o = rel(p_occ, p_ref)
        tag = misfit + (" log" if logd else "")
        print(f"[pieces {tag:13s}] H_m {e_h:.1e} | gradient via J {e_g:.1e} "
              f"| direct vs CG({it}) {e_s:.1e} | occam step vs direct at its beta {e_o:.1e}")
        assert e_o < 1e-8, f"Occam step is not the exact step at its beta ({e_o})"
        assert e_h < 1e-12, f"H_m disagrees with autograd ({e_h})"
        assert e_g < 1e-10, f"gradient via J disagrees with autograd ({e_g})"
        assert e_s < 1e-6, f"direct and converged CG steps differ ({e_s})"


def run(opt, mesh, eng, dat, sim, beta0=1e3, schedule=True):
    from simpeg import directives, inversion

    class Track(directives.InversionDirective):
        def initialize(self):
            self.f = []

        def endIter(self):
            self.f.append((float(self.invProb.beta), float(self.opt.f)))

    prob, m0 = inv_problem(mesh, eng, dat, sim, opt, beta=beta0)
    tr = Track()
    dirs = ([directives.BetaSchedule(coolingFactor=2.0, coolingRate=1)]
            if schedule else [])
    inv = inversion.BaseInversion(prob, directiveList=dirs + [
        directives.TargetMisfit(chifact=1.0), tr])
    with contextlib.redirect_stdout(io.StringIO()):
        m = inv.run(m0)
    return m, prob.phi_d / dat.dobs.size, len(tr.f), float(prob.beta)


def inversions():
    from dctorch import TorchGaussNewton, TorchLBFGS

    mesh, eng, dat, sim = problem()
    out = {}
    for name, opt in [
        ("gn explicit", TorchGaussNewton(engine=eng, maxIter=20, jacobian="explicit",
                                         beta_search=False)),
        ("gn matrix-free", TorchGaussNewton(engine=eng, maxIter=20,
                                            jacobian="matrix_free", beta_search=False)),
        ("gn direct", TorchGaussNewton(engine=eng, maxIter=20, jacobian="explicit",
                                       inner="direct", beta_search=False)),
        ("lbfgs", TorchLBFGS(engine=eng, maxIter=20, inner_maxiter=30)),
    ]:
        m, chi2, nit, _ = run(opt, mesh, eng, dat, sim)
        out[name] = m
        print(f"[inversion {name:14s}] chi2 {chi2:.2f} in {nit} iterations")
        assert chi2 <= 1.0, f"{name}: target misfit not reached ({chi2:.2f})"
    e = np.linalg.norm(out["gn explicit"] - out["gn matrix-free"]) \
        / np.linalg.norm(out["gn explicit"])
    print(f"[inversion paths  ] explicit vs matrix-free model: {e:.1e}")
    assert e < 1e-4, f"explicit and matrix-free paths diverged ({e})"

    # (f) Occam: no schedule, beta chosen in each step, from very different beta0
    finals = []
    for b0 in (1e5, 10.0):
        opt = TorchGaussNewton(engine=eng, maxIter=20, jacobian="explicit",
                               beta_search=True)
        _, chi2, nit, b_end = run(opt, mesh, eng, dat, sim, beta0=b0, schedule=False)
        finals.append(b_end)
        print(f"[occam beta0={b0:7.0e}] chi2 {chi2:.2f} in {nit} iterations | "
              f"final beta {b_end:.3g}")
        assert 0.6 <= chi2 <= 1.0, f"Occam ended at chi2 {chi2:.2f}"
        assert opt.counts["occam"] == nit
    spread = max(finals) / min(finals)
    print(f"[occam            ] final beta from beta0 10^4 apart: x{spread:.2f}")
    assert spread < 3.0, f"Occam's final beta depends on beta0 (x{spread:.2f})"


def main():
    pieces()
    inversions()
    print(f"GAUSS-NEWTON VALIDADO ({DEV})")


if __name__ == "__main__":
    main()
