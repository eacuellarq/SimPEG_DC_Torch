"""
Wavenumber quadrature for the 2.5-D transform, fitted where the data live.

The 2.5-D forward solves one 2-D problem per wavenumber k and sums them,
u = sum_k w_k u_k. The points and weights are chosen so that the same sum
reproduces the 3-D point-source potential,

    sum_k g_k K_0(k r) = 1 / r,      w_k = g_k / 2,

over the distances r that matter. Stock SimPEG 0.25 (`Simulation2DNodal`)
fits this with `scipy.optimize.minimize` over r from min(edge length)/4 to
4 x the mesh extent, solving for g by the normal equations. Two things go
wrong with that:

- the range is set by the MESH, so on a padded mesh most of the fitting
  effort goes to distances no electrode pair ever spans;
- at larger nky the fit becomes ill-conditioned (cond(A^T A) passes 1e17 at
  nky = 31 on a typical line), `minimize` reports failure, and SimPEG falls
  back -- with only a warning -- to a trapezoid rule on logspace(-4, 1) that
  ignores the geometry entirely. Measured: 2.46 % error in 1/r, which is the
  "nky = 31 degrades to 2.47 %" of notes/HANDOFF_2026-09-27.

`k0_quadrature` fits over the electrode distances of the survey instead (what
RES2DINV does), solves for g by non-negative least squares, optimizes the
log-wavenumbers with `least_squares` on the residual vector, grows the points
one at a time so more of them never fit worse, and REPORTS the error it
reached on a dense grid -- a poor fit is a number you can see, not a silent
fallback.

Why non-negative: an unconstrained fit at nky >= 21 reaches a tiny ANALYTIC
error with weights of alternating sign (sum|w| = 332 against sum w = 0.86 on
a 5-120 m line), and those weights multiply the discretization error of every
2-D solve by the same factor -- the data got worse as the transform got
better. Positive weights are what a quadrature of a positive integrand
should have; `amplification(w)` = sum|w| / sum w reports it (1 when positive;
SimPEG's own fit has a few negative weights at nky = 11).
"""
import warnings

import numpy as np


def electrode_distance_range(survey):
    """(r_min, r_max) over every current-electrode / potential-electrode pair
    of the survey: the distances the transform is actually evaluated at."""
    rmin, rmax = np.inf, 0.0
    for src in survey.source_list:
        locs = [np.atleast_2d(np.asarray(src.location_a, float))]
        if getattr(src, "location_b", None) is not None:
            locs.append(np.atleast_2d(np.asarray(src.location_b, float)))
        cur = np.vstack(locs)[:, :2]
        cur = cur[np.all(np.isfinite(cur), axis=1)]   # a pole source's B is nan
        for rx in src.receiver_list:
            pots = [np.atleast_2d(np.asarray(rx.locations_m if hasattr(rx, "locations_m")
                                             else rx.locations, float))]
            if getattr(rx, "locations_n", None) is not None:
                pots.append(np.atleast_2d(np.asarray(rx.locations_n, float)))
            pot = np.vstack(pots)[:, :2]
            pot = pot[np.all(np.isfinite(pot), axis=1)]
            r = np.linalg.norm(cur[:, None, :] - pot[None, :, :], axis=-1).ravel()
            r = r[r > 1e-9]
            if r.size:
                rmin, rmax = min(rmin, r.min()), max(rmax, r.max())
    if not np.isfinite(rmin):
        raise ValueError("the survey has no separated current/potential electrode pair")
    return float(rmin), float(rmax)


def amplification(weights):
    """sum|w| / |sum w|: how much the quadrature magnifies errors of the
    individual 2-D solves. 1 for positive weights; a fit with cancelling
    weights can reproduce K_0 analytically and still multiply the
    discretization error of each wavenumber by hundreds."""
    w = np.asarray(weights, float)
    return float(np.abs(w).sum() / abs(w.sum()))


def _nnls(A, b):
    """Non-negative least squares, robust to the K_0 columns spanning many
    orders of magnitude: columns are normalized first, and if scipy's active
    set runs out of iterations the bounded solver takes over."""
    from scipy.optimize import lsq_linear, nnls

    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    As = A / scale
    try:
        x = nnls(As, b, maxiter=50 * A.shape[1])[0]
    except RuntimeError:
        x = lsq_linear(As, b, bounds=(0.0, np.inf), method="bvls").x
    return x / scale


def _design(r, k):
    from scipy.special import k0
    return r[:, None] * k0(r[:, None] * k[None, :])


def transform_error(points, weights, r_min, r_max, n=400):
    """Max relative error of sum_k 2 w_k K_0(k r) against 1/r on [r_min, r_max]."""
    r = np.logspace(np.log10(r_min), np.log10(r_max), n)
    return float(np.abs(_design(r, np.asarray(points)) @ (2.0 * np.asarray(weights)) - 1.0).max())


def k0_quadrature(r_min, r_max, nky, margin=2.0, n_samples=200, n_start=4):
    """Points and weights (SimPEG's convention, w = g/2) fitted on
    [r_min / margin, r_max * margin].

    The fit is non-convex, and asked for more points than the range needs a
    direct optimization stalls in poor minima -- more points then mean a
    WORSE transform. So the points are grown one at a time (continuation):
    each new point goes into the widest gap of the previous optimum, which is
    the starting guess for the next fit. Adding a point can then only help,
    up to round-off.

    Returns (points, weights, info) with info = {success, max_rel_err,
    r_range, err_by_n, n_used, amplification}; errors are measured on
    [r_min, r_max], the range the data use. Points NNLS gives zero weight are
    dropped (n_used <= nky): they would cost a solve and add nothing.
    """
    from scipy.optimize import least_squares

    rs = np.logspace(np.log10(r_min / margin), np.log10(r_max * margin), n_samples)
    e = np.ones_like(rs)

    def g_of(lk):
        return _nnls(_design(rs, 10.0 ** lk), e)

    def resid(lk):
        A = _design(rs, 10.0 ** lk)
        return A @ _nnls(A, e) - e

    def fit(lk_init):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = least_squares(resid, lk_init, method="trf", xtol=1e-12,
                                ftol=1e-12, max_nfev=200 * lk_init.size)
        return np.sort(out.x), bool(out.success)

    lo, hi = -np.log10(rs.max()), -np.log10(rs.min())   # k ~ 1/r
    n0 = min(n_start, nky)
    lk, ok = fit(np.linspace(lo, hi, n0))
    err_by_n = {n0: transform_error(10.0 ** lk, g_of(lk) / 2.0, r_min, r_max)}
    for n in range(n0 + 1, nky + 1):
        edges = np.r_[lo - 0.5, lk, hi + 0.5]            # the ends are gaps too
        j = int(np.argmax(np.diff(edges)))
        lk, ok = fit(np.sort(np.r_[lk, 0.5 * (edges[j] + edges[j + 1])]))
        err_by_n[n] = transform_error(10.0 ** lk, g_of(lk) / 2.0, r_min, r_max)

    points = 10.0 ** lk
    weights = g_of(lk) / 2.0
    keep = weights > 0                     # NNLS may switch points off entirely
    if keep.sum() < keep.size:
        points, weights = points[keep], weights[keep]
    info = {"success": ok and bool(np.all(np.isfinite(weights))),
            "max_rel_err": err_by_n[nky], "r_range": (float(r_min), float(r_max)),
            "err_by_n": err_by_n, "n_used": int(points.size),
            "amplification": amplification(weights)}
    return points, weights, info
