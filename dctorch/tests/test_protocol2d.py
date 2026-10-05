"""
Validación Fase 3: inversión completa por el protocolo SimPEG 0.25 stock con
TorchLBFGS(engine=TorchDC2D) — (A) L2 + directives oficiales de beta/target,
(B) Sparse + UpdateIRLS OFICIAL del 0.25 (nuestro port se retiró).
Correr desde la raíz del repo:  python -m dctorch.tests.test_protocol2d
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time

import numpy as np
import torch

from dctorch.tests.test_simulation2d import build_problem


def make_inversion(reg_kind):
    from simpeg import (data, data_misfit, directives, inverse_problem,
                        inversion, regularization)
    from dctorch import TorchDC2D, TorchLBFGS

    mesh, survey, sim, m_true = build_problem()
    eng = TorchDC2D(mesh, survey, nky=11, device="cuda", quadrature="simpeg")

    rng = np.random.default_rng(7)
    d_clean = sim.dpred(m_true)
    std = 0.03 * np.abs(d_clean) + 1e-6
    dobs = d_clean + std * rng.standard_normal(len(d_clean))
    dat = data.Data(survey, dobs=dobs, standard_deviation=std)
    dmis = data_misfit.L2DataMisfit(data=dat, simulation=sim)

    m0 = np.log(100.0) * np.ones(mesh.nC)
    if reg_kind == "l2":
        reg = regularization.WeightedLeastSquares(
            mesh, alpha_s=0.16, alpha_x=1.0, alpha_y=1.0,
            reference_model=m0)
        dirs = [directives.BetaEstimate_ByEig(beta0_ratio=100.0),
                directives.BetaSchedule(coolingFactor=2.0, coolingRate=1),
                directives.TargetMisfit(chifact=1.0)]
        maxiter = 15
    else:
        reg = regularization.Sparse(
            mesh, norms=[0.0, 2.0, 2.0], alpha_s=0.16, alpha_x=1.0,
            alpha_y=1.0, reference_model=m0)
        dirs = [directives.UpdateIRLS(
                    cooling_factor=3.0, cooling_rate=1, chifact_start=1.0,
                    chifact_target=1.0, max_irls_iterations=8,
                    percentile=90.0, f_min_change=1e-4),
                directives.BetaEstimate_ByEig(beta0_ratio=10.0)]
        maxiter = 25

    opt = TorchLBFGS(engine=eng, maxIter=maxiter, inner_maxiter=25,
                     inner_restarts=2)
    prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    inv = inversion.BaseInversion(prob, directiveList=dirs)
    blk = ((np.abs(mesh.cell_centers[:, 0] - 5) < 8)
           & (mesh.cell_centers[:, 1] < -3) & (mesh.cell_centers[:, 1] > -10))
    return inv, prob, m0, m_true, survey.nD, blk


def run(reg_kind):
    t0 = time.perf_counter()
    inv, prob, m0, m_true, nD, blk = make_inversion(reg_kind)
    m_rec = inv.run(m0)
    dt = time.perf_counter() - t0
    chi2 = prob.phi_d / nD                 # 0.25: phi_d = ||Wr||^2, target nD
    # metrica fisica: el bloque conductor (verdad log10=1) debe aparecer bajo
    # el fondo (log100=4.6) DONDE esta, y el fondo lejano quedarse quieto
    dip_blk = float(np.log(100.0) - m_rec[blk].mean())
    drift_bg = float(np.abs(m_rec[~blk] - np.log(100.0)).mean())
    print(f"[{reg_kind:6s}] chi2 {chi2:.3f} | caida en el bloque "
          f"{dip_blk:.2f} (verdad 2.30) | deriva del fondo {drift_bg:.2f} "
          f"| {dt:.0f}s")
    assert chi2 < 2.0, f"no ajusto: chi2 {chi2}"
    assert dip_blk > 0.8, f"no recupero el conductor: {dip_blk}"
    assert drift_bg < 0.6, f"fondo distorsionado: {drift_bg}"
    return chi2


if __name__ == "__main__":
    run("l2")
    run("sparse")
    print("FASE 3 VALIDADA")
