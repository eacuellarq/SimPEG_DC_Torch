"""Does choosing beta automatically beat cooling it on a schedule?

The production recipe cools beta by a fixed factor and stops when chi2 reaches
its target. That is a crude homotopy, and it was measured not to compensate for
forward error. The numerical-analysis alternative is to pick the regularization
parameter INSIDE each linearized step, from quantities the step itself produces
-- the hybrid-projection family surveyed by Chung and Gazzola (SIAM Review 66,
205-284, 2024). This measures whether the choice of rule matters at all, before
anyone builds the machinery that would make it cheap.

What is compared, all inside the same Gauss-Newton loop:
  cool        beta_k = beta_0 / 2^k, the schedule
  discrepancy beta such that the predicted misfit equals its target (chi2 = 1)
  gcv         beta minimizing generalized cross-validation
  fixed       beta_0 throughout, as a control

The honest caveat about cost. The efficient form of these methods never forms J:
it runs k Golub-Kahan steps, each needing one J v and one J^T u, and dctorch has
only the second. So here J is formed explicitly by nD calls to `Jtvec`, which
costs nD gradient-equivalents per Gauss-Newton step and is only viable because
this case is small (93 data). The question being answered is whether the RULE
pays, not whether this implementation of it is fast. If the rule pays, that is
the argument for building the forward products; if it does not, that work is
saved.

The algebra. With B = W J, r = W (d - dobs), q = m - m_ref and the thin SVD
B = U S V^T, the Tikhonov step on the model is

    delta(beta) = -V [ (s u + beta a) / (s^2 + beta) ] - (q - V a)

with u = U^T r, a = V^T q, and its predicted misfit is

    || B delta + r ||^2 = sum_i [ beta (u_i - s_i a_i) / (s_i^2 + beta) ]^2

monotone in beta, so the discrepancy principle has one root. The influence-matrix
trace is sum_i s_i^2 / (s_i^2 + beta), which gives GCV in closed form. Both
formulas are checked against a direct dense solve at run time.

Env
  SEEDS (7,11,23)  MAXIT (25)  CHUNK (8)  NOISE (0.03)  DEV
  SAVE   path of an .npz with the first seed's models and trajectories

  python -u hybrid_beta_2d.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from haar_inversion_2d import build                                # noqa: E402
from dctorch import TorchDC2D                                     # noqa: E402

SEEDS = [int(s) for s in os.environ.get("SEEDS", "7,11,23").split(",")]
MAXIT = int(os.environ.get("MAXIT", "25"))
CHUNK = int(os.environ.get("CHUNK", "8"))
NOISE = float(os.environ.get("NOISE", "0.03"))
DEV = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
SAVE = os.environ.get("SAVE", "")       # .npz of the first seed, for the figure
DT = torch.float64
NKY = 11
RULES = ("cool", "discrepancy", "gcv", "fixed")


def jacobian(eng, m, nD):
    """J = d(dpred)/dm by nD reverse passes. Costs nD gradient-equivalents."""
    rows = []
    e = torch.zeros(nD, dtype=DT, device=DEV)
    for i in range(nD):
        e.zero_()
        e[i] = 1.0
        rows.append(eng.Jtvec(m, e, chunk=CHUNK or None))
    return torch.stack(rows, 0)                        # (nD, nM)


def misfit_of(beta, u, s, a):
    """||B delta(beta) + r||^2, from the SVD only."""
    return float((((beta * (u - s * a)) / (s ** 2 + beta)) ** 2).sum())


def step_of(beta, U, s, V, u, a, q):
    coef = (s * u + beta * a) / (s ** 2 + beta)
    return -(V @ coef) - (q - V @ a)


def pick_beta(rule, k, beta0, U, s, V, u, a, q, nD):
    if rule == "fixed":
        return beta0
    if rule == "cool":
        return beta0 / (2.0 ** k)
    lo, hi = 1e-14 * float(s[0] ** 2), 1e6 * float(s[0] ** 2)
    if rule == "discrepancy":
        target = float(nD)                             # chi2 = 1
        if misfit_of(hi, u, s, a) < target:            # cannot be reached
            return hi
        if misfit_of(lo, u, s, a) > target:
            return lo
        for _ in range(80):                            # monotone: bisect on log
            mid = np.sqrt(lo * hi)
            if misfit_of(mid, u, s, a) < target:
                lo = mid
            else:
                hi = mid
        return np.sqrt(lo * hi)
    # gcv: minimize nD * ||res||^2 / trace(I - influence)^2 on a log grid
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 200))
    best, bbeta = np.inf, beta0
    for b in grid:
        num = nD * misfit_of(b, u, s, a)
        den = float((b / (s ** 2 + b)).sum()) ** 2
        v = num / max(den, 1e-300)
        if v < best:
            best, bbeta = v, b
    return bbeta


def run(rule, eng, dobs, w, m_ref, nD, beta0, m_true):
    m = m_ref.clone()
    hist = []                       # (grad-equivalents, chi2, beta) per step
    nev = 0
    t0 = time.perf_counter()
    chi2 = np.inf
    for k in range(MAXIT):
        with torch.no_grad():
            d = eng.dpred(m)
        nev += 1
        r = w * (d - dobs)
        chi2 = float((r ** 2).sum()) / nD
        if chi2 <= 1.0:
            break
        J = jacobian(eng, m, int(nD))
        nev += int(nD)
        B = w.unsqueeze(1) * J
        U, s, Vh = torch.linalg.svd(B, full_matrices=False)
        V = Vh.T
        u, q = U.T @ r, m - m_ref
        a = V.T @ q
        beta = pick_beta(rule, k, beta0, U, s, V, u, a, q, nD)
        hist.append((nev, chi2, float(beta)))

        if k == 0:                                     # check the algebra once
            delta = step_of(beta, U, s, V, u, a, q)
            N = B.T @ B + beta * torch.eye(m.numel(), dtype=DT, device=DEV)
            ref = torch.linalg.solve(N, -(B.T @ r + beta * q))
            err = float(torch.linalg.norm(delta - ref) / torch.linalg.norm(ref))
            pm = misfit_of(beta, u, s, a)
            pm_ref = float(torch.linalg.norm(B @ delta + r) ** 2)
            print(f"    [{rule:11s}] svd step vs dense solve {err:.1e} | "
                  f"predicted misfit formula vs direct "
                  f"{abs(pm - pm_ref) / pm_ref:.1e}")

        delta = step_of(beta, U, s, V, u, a, q)
        acc = False                                    # damped Gauss-Newton
        t = 1.0
        for _ in range(8):
            with torch.no_grad():
                dn = eng.dpred(m + t * delta)
            nev += 1
            c2n = float(((w * (dn - dobs)) ** 2).sum()) / nD
            if c2n < chi2:
                m, chi2, acc = m + t * delta, c2n, True
                break
            t *= 0.5
        if not acc:
            break
    hist.append((nev, chi2, hist[-1][2] if hist else beta0))
    corr = float(np.corrcoef(m.cpu().numpy(), m_true)[0, 1])
    return (chi2, k + 1, nev, time.perf_counter() - t0, corr,
            m.cpu().numpy(), np.array(hist))


def main():
    mesh, survey, m_true, body = build()
    nC, nD = mesh.nC, float(survey.nD)
    print(f"mesh nC={nC:,} nD={survey.nD} nS={len(survey.source_list)} | {DEV} "
          f"| chunk {CHUNK} | noise {NOISE:.0%} | Gauss-Newton, max {MAXIT} steps")
    print(f"cost note: one Gauss-Newton step here = 1 + {survey.nD} + line search "
          f"gradient-equivalents, because J is formed by {survey.nD} Jtvec calls")
    eng = TorchDC2D(mesh, survey, nky=NKY, device=DEV, dtype=DT)
    mt = torch.tensor(m_true, dtype=DT, device=DEV)
    with torch.no_grad():
        d_clean = eng.dpred(mt).cpu().numpy()
    m_ref = torch.tensor(np.log(100.0) * np.ones(nC), dtype=DT, device=DEV)

    res = {k: [] for k in RULES}
    saved = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        dobs = torch.tensor(d_clean * (1 + NOISE * rng.standard_normal(d_clean.size)),
                            dtype=DT, device=DEV)
        w = torch.tensor(1.0 / (NOISE * np.abs(d_clean) + 1e-12), dtype=DT, device=DEV)
        # beta0 as the production recipe scales it: from the spectrum at m_ref
        J0 = jacobian(eng, m_ref, int(nD))
        s0 = torch.linalg.svdvals(w.unsqueeze(1) * J0)
        beta0 = float(s0[0] ** 2) / 100.0
        print(f"  seed {seed:3d} | beta0 {beta0:.3e}")
        for rule in RULES:
            chi2, it, nev, sec, corr, m_fin, hist = run(
                rule, eng, dobs, w, m_ref, nD, beta0, m_true)
            res[rule].append((chi2, it, nev, sec, corr))
            if SAVE and seed == SEEDS[0]:
                saved[f"m_{rule}"] = m_fin
                saved[f"hist_{rule}"] = hist
                saved[f"stat_{rule}"] = np.array([chi2, it, nev, corr])
            print(f"    {rule:11s} chi2 {chi2:8.3f} | {it:2d} GN steps | "
                  f"{nev:5d} grad-equiv | corr {corr:6.3f} | {sec:5.2f} s",
                  flush=True)

    print(f"\n{'rule':12s} {'hit':>5s} {'median GN':>10s} {'median g-eq':>12s} "
          f"{'median chi2':>12s} {'median corr':>12s}")
    for rule in RULES:
        r = res[rule]
        print(f"{rule:12s} {sum(1 for x in r if x[0] <= 1.0):>2d}/{len(r):<2d} "
              f"{np.median([x[1] for x in r]):10.0f} "
              f"{np.median([x[2] for x in r]):12.0f} "
              f"{np.median([x[0] for x in r]):12.3f} "
              f"{np.median([x[4] for x in r]):12.3f}")


    if SAVE:
        saved["m_true"] = m_true
        saved["seed"] = np.array([SEEDS[0]])
        np.savez(SAVE, **saved)
        print(f"\nmodels and trajectories of seed {SEEDS[0]} -> {SAVE}")


if __name__ == "__main__":
    main()
