"""synth3d_survey_explore.py — trade lateral resolution for depth at a FIXED data budget.

Each candidate rescales the electrode spacing and redistributes the receiver
march inside the same 2.8 x 2.0 km footprint, with the source count held at the
current 184 and the data count held at or below the current 13,482. Geometry
only (no mesh, no solve): data, sources, electrodes, Edwards median depth over
a flat half-space, how much of the survey still senses the shallow units, and
the weakest signals over a 500 ohm-m half-space.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

import synth3d_common as S

BUDGET_D, BUDGET_S = 13482, 184


def run(name, a, src_len, src_step, inline, off, tie_step=None):
    S.LINE_X = np.arange(-1400.0, 1400.0 + 1e-6, a)
    S.TIE_Y = np.arange(-1000.0, 1000.0 + 1e-6, a)
    S.SRC_LEN, S.SRC_STEP = src_len, src_step
    S.INLINE_TIERS, S.OFFLINE_TIERS = inline, off
    geo = S.survey_geometry(drop_null=False)   # the exploration scores the full layout
    A, B, M, N = [], [], [], []
    for sa, sb, Ms, Ns, _ in geo:
        for m, n in zip(Ms, Ns):
            A.append(sa); B.append(sb); M.append(m); N.append(n)
    A, B, M, N = map(np.array, (A, B, M, N))
    bad = S.edwards_ill_defined(A, B, M, N)
    z = S.edwards_median_depth(A, B, M, N)[~bad]      # depth stats: well-defined data only
    r = lambda p, q: np.linalg.norm(p - q, axis=1)  # noqa: E731
    v = 500.0 / (2 * np.pi) * np.abs(1 / r(A, M) - 1 / r(A, N) - 1 / r(B, M) + 1 / r(B, N))
    ok = "" if (len(A) <= BUDGET_D and len(geo) <= BUDGET_S) else "  OVER BUDGET"
    print(f"{name:<34} a={a:>4.0f} AB={src_len*a:>4.0f} | nS {len(geo):3d} nD {len(A):6d} "
          f"ill {bad.sum():4d} | z50 med {np.median(z):4.0f} p90 {np.percentile(z, 90):4.0f} "
          f"| >=210 {np.mean(z >= 210)*100:4.1f}% >=320 {np.mean(z >= 320)*100:4.1f}% "
          f">=430 {np.mean(z >= 430)*100:4.1f}% | <=60 {np.mean(z <= 60)*100:4.1f}% "
          f"| |V/I| p1 {np.percentile(v, 1):.1e}{ok}", flush=True)


print("current design, for reference")
run("C0 current (50 m)", 50, 2, 4, ((1, 8), (2, 4), (4, 3)),
    {1: ((1, 8), (2, 4)), 2: ((2, 8),)})

print("\nelectrode spacing x2 = 100 m")
run("C1 same tiers", 100, 2, 2, ((1, 8), (2, 4), (4, 3)),
    {1: ((1, 8), (2, 4)), 2: ((2, 8),)})
run("C2 AB 100, same tiers", 100, 1, 2, ((1, 8), (2, 4), (4, 3)),
    {1: ((1, 8), (2, 4)), 2: ((2, 8),)})
run("C3 longer inline, short offline", 100, 2, 2, ((1, 6), (2, 6), (4, 4)),
    {1: ((1, 6), (2, 4)), 2: ((2, 6),)})
run("C4 offline to 3rd line", 100, 2, 2, ((1, 6), (2, 4), (4, 3)),
    {1: ((1, 6), (2, 3)), 2: ((2, 5),), 3: ((2, 5),)})
run("C5 all 200 m offline", 100, 2, 2, ((1, 6), (2, 4), (4, 3)),
    {1: ((2, 7),), 2: ((2, 7),), 3: ((2, 5),)})

print("\n100 m, spending the remaining data budget on depth")
run("C8 C1 + 3rd offline line", 100, 2, 2, ((1, 8), (2, 4), (4, 3)),
    {1: ((1, 8), (2, 4)), 2: ((2, 8),), 3: ((2, 5),)})
run("C9 C5 + longer inline", 100, 2, 2, ((1, 6), (2, 4), (4, 4)),
    {1: ((2, 8),), 2: ((2, 8),), 3: ((2, 6),)})
run("C10 C8 + 4th line, 400 m dipoles", 100, 2, 2, ((1, 8), (2, 4), (4, 3)),
    {1: ((1, 8), (2, 4)), 2: ((2, 8),), 3: ((2, 5),), 4: ((4, 3),)})
run("C11 C1 + 3rd/4th, 400 m dipoles", 100, 2, 2, ((1, 8), (2, 4), (4, 3)),
    {1: ((1, 8), (2, 4)), 2: ((2, 8),), 3: ((4, 4),), 4: ((4, 4),)})

print("\nelectrode spacing x3 / x4")
run("C6 150 m", 150, 2, 2, ((1, 6), (2, 4), (4, 2)),
    {1: ((1, 5), (2, 3)), 2: ((2, 5),)})
run("C7 200 m", 200, 1, 1, ((1, 6), (2, 3), (4, 2)),
    {1: ((1, 5), (2, 2)), 2: ((1, 5), (2, 2))})
