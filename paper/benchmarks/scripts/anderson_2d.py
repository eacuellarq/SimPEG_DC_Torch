"""Anderson acceleration against L-BFGS, counted in gradient evaluations.

Why this and not something cleverer first: Anderson acceleration needs no extra
PDE solve, no Jacobian and no change to the engine. It is a small least-squares
problem over the history of iterates, and the seismic literature reports it
beating both L-BFGS and non-linear CG when the count is gradient evaluations
(Geophysics 86(1), R99) -- which is the metric that decides wall-clock here,
since one gradient is one factorization plus one solve per source batch.

The method. Write the descent as a fixed-point map G(x) = x - a grad(x), whose
residual is f(x) = G(x) - x. Anderson replaces the plain step by the combination
of the last m iterates whose residual is smallest:

    gamma = argmin || f_k - dF gamma ||,    x_{k+1} = G(x_k) - (dX + dF) gamma

with dX and dF the differences of the stored iterates and residuals. The
least-squares problem is m x m with m <= 10, so it costs nothing next to a solve.

Safeguarded, because unsafeguarded Anderson can diverge: an accelerated step is
accepted only if it lowers the objective; otherwise the plain gradient step is
taken and the history is dropped. Those rejected trials are counted as
evaluations, so the comparison is not flattered.

Same problem, objective, starting model and source batching as
`haar_inversion_2d.py`, so the L-BFGS column here is the same baseline measured
there. Several noise realizations by default -- one seed misled us on the Haar
question, and the lesson was cheap to learn twice.

Env
  SEEDS   comma-separated noise seeds (default 7,11,23,31,47)
  MEM     Anderson history length, comma-separated (default 5,10)
  MAXEV   gradient-evaluation budget per method (default 120)
  LBFGS_MAXITER  inner iterations per L-BFGS step call (default 1; production is 20)
  CHUNK   sources solved together (default 8)
  BETA    weight of the smallness term (default 0.01)
  NOISE   relative noise (default 0.03)
  DEV     cuda | cpu

  python -u anderson_2d.py
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
from dctorch import TorchDC2D                                      # noqa: E402

SEEDS = [int(s) for s in os.environ.get("SEEDS", "7,11,23,31,47").split(",")]
MEMS = [int(s) for s in os.environ.get("MEM", "5,10").split(",")]
MAXEV = int(os.environ.get("MAXEV", "120"))
CHUNK = int(os.environ.get("CHUNK", "8"))
BETA = float(os.environ.get("BETA", "0.01"))
NOISE = float(os.environ.get("NOISE", "0.03"))
DEV = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
# inner iterations per torch.optim.LBFGS.step call. Production (TorchLBFGS)
# uses 20; 1 lets the chi2 target be checked every iteration. Both are tried,
# because a handicapped baseline would invalidate the comparison.
LB_MAXIT = int(os.environ.get("LBFGS_MAXITER", "1"))
DT = torch.float64
NKY = 11
TARGET = 1.0


class Objective:
    """phi_d + beta * smallness, and its gradient. Counts every evaluation."""

    def __init__(self, eng, dobs, w, m_ref, nD):
        self.eng, self.dobs, self.w = eng, dobs, w
        self.m_ref, self.nD = m_ref, nD
        self.n = 0

    def __call__(self, m):
        self.n += 1
        fd, g = self.eng.misfit_and_grad(m, self.dobs, self.w,
                                         chunk=CHUNK or None)
        r = m - self.m_ref
        return fd + BETA * float((r ** 2).sum()), g + 2.0 * BETA * r, fd / self.nD


def run_lbfgs(obj, m0):
    """The current production first-order method, as measured elsewhere."""
    p = m0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], max_iter=LB_MAXIT, history_size=10,
                            line_search_fn="strong_wolfe")
    st = {}

    def closure():
        opt.zero_grad()
        f, g, chi2 = obj(p.detach())
        p.grad = g
        st["chi2"] = chi2
        return f

    t0 = time.perf_counter()
    while obj.n < MAXEV:
        opt.step(closure)
        if st["chi2"] <= TARGET:
            break
    return st["chi2"], obj.n, time.perf_counter() - t0, p.detach()


def run_gd(obj, m0, alpha):
    """Plain gradient descent at the same step: the control."""
    x = m0.clone()
    f, g, chi2 = obj(x)
    t0 = time.perf_counter()
    while obj.n < MAXEV and chi2 > TARGET:
        x = x - alpha * g
        f, g, chi2 = obj(x)
    return chi2, obj.n, time.perf_counter() - t0, x


def run_anderson(obj, m0, alpha, mem, line_search=False):
    """Safeguarded Anderson acceleration on G(x) = x - alpha * grad(x).

    With line_search, the accelerated direction is backtracked instead of being
    accepted or rejected whole. Without it, Anderson competes against an L-BFGS
    that line-searches, which would not be a fair fight.
    """
    x = m0.clone()
    f, g, chi2 = obj(x)
    dX, dF = [], []
    x_prev = res_prev = None
    t0 = time.perf_counter()
    while obj.n < MAXEV and chi2 > TARGET:
        res = -alpha * g                              # f(x) = G(x) - x
        if x_prev is not None:
            dX.append(x - x_prev)
            dF.append(res - res_prev)
            if len(dX) > mem:
                dX.pop(0)
                dF.pop(0)
        x_prev, res_prev = x.clone(), res.clone()

        step = res                                    # the plain step, m = 0
        if dX:
            DF = torch.stack(dF, 1)                   # (nM, m)
            DX = torch.stack(dX, 1)
            G = DF.T @ DF
            lam = 1e-10 * torch.diagonal(G).sum() / G.shape[0]
            gam = torch.linalg.solve(
                G + lam * torch.eye(G.shape[0], dtype=DT, device=x.device),
                DF.T @ res)
            step = res - (DX + DF) @ gam

        if line_search:
            ok = False
            s = 1.0
            for _ in range(8):
                f_new, g_new, chi2_new = obj(x + s * step)
                if f_new < f:
                    x, f, g, chi2 = x + s * step, f_new, g_new, chi2_new
                    ok = True
                    break
                s *= 0.5
                if obj.n >= MAXEV:
                    break
            if ok:
                continue
            dX.clear()
            dF.clear()
            x_prev = res_prev = None
            continue

        f_new, g_new, chi2_new = obj(x + step)
        if f_new < f:                                 # accept
            x, f, g, chi2 = x + step, f_new, g_new, chi2_new
        else:                                         # reject: plain step, reset
            dX.clear()
            dF.clear()
            f_new, g_new, chi2_new = obj(x + res)
            x, f, g, chi2 = x + res, f_new, g_new, chi2_new
            x_prev = res_prev = None
    return chi2, obj.n, time.perf_counter() - t0, x


def initial_step(obj, m0):
    """Backtracking at the start, so every method shares one honest step."""
    f0, g0, _ = obj(m0)
    a = 1.0 / float(torch.linalg.norm(g0))
    for _ in range(40):
        f1, _, _ = obj(m0 - a * g0)
        if f1 < f0:
            return a
        a *= 0.5
    return a


def main():
    mesh, survey, m_true, body = build()
    nC, nD = mesh.nC, float(survey.nD)
    print(f"mesh nC={nC:,} nD={survey.nD} nS={len(survey.source_list)} nky={NKY} "
          f"| {DEV} | chunk {CHUNK} | beta {BETA:g} | noise {NOISE:.0%} "
          f"| budget {MAXEV} gradient evaluations")
    eng = TorchDC2D(mesh, survey, nky=NKY, device=DEV, dtype=DT)
    mt = torch.tensor(m_true, dtype=DT, device=DEV)
    with torch.no_grad():
        d_clean = eng.dpred(mt).cpu().numpy()
    m_ref = torch.tensor(np.log(100.0) * np.ones(nC), dtype=DT, device=DEV)

    names = (["L-BFGS", "grad descent"]
             + [f"Anderson m={m}" for m in MEMS]
             + [f"Anderson+ls m={m}" for m in MEMS])
    res = {k: [] for k in names}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        dobs = torch.tensor(d_clean * (1 + NOISE * rng.standard_normal(d_clean.size)),
                            dtype=DT, device=DEV)
        w = torch.tensor(1.0 / (NOISE * np.abs(d_clean) + 1e-12), dtype=DT, device=DEV)

        probe = Objective(eng, dobs, w, m_ref, nD)
        alpha = initial_step(probe, m_ref)

        line = [f"  seed {seed:3d} | step {alpha:.2e} |"]
        for name in names:
            obj = Objective(eng, dobs, w, m_ref, nD)
            if name == "L-BFGS":
                out = run_lbfgs(obj, m_ref)
            elif name == "grad descent":
                out = run_gd(obj, m_ref, alpha)
            else:
                out = run_anderson(obj, m_ref, alpha, int(name.split("=")[1]),
                                   line_search=name.startswith("Anderson+ls"))
            chi2, nev, sec, m_fin = out
            hit = chi2 <= TARGET
            corr = float(np.corrcoef(m_fin.cpu().numpy(), m_true)[0, 1])
            res[name].append((chi2, nev, sec, hit, corr))
            line.append(f" {name} {'chi2 %.2f' % chi2} in {nev:3d} ev"
                        f"{'' if hit else ' (no target)'};")
        print("".join(line), flush=True)

    print(f"\n{'method':16s} {'target hit':>10s} {'median ev':>10s} "
          f"{'median chi2':>12s} {'median corr':>12s} {'median s':>9s}")
    for name in names:
        r = res[name]
        hits = sum(1 for x in r if x[3])
        ev = np.median([x[1] for x in r])
        c2 = np.median([x[0] for x in r])
        co = np.median([x[4] for x in r])
        se = np.median([x[2] for x in r])
        print(f"{name:16s} {hits:>4d}/{len(r):<5d} {ev:10.0f} {c2:12.3f} "
              f"{co:12.3f} {se:9.2f}")


if __name__ == "__main__":
    main()
