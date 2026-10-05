"""synth3d_view3d.py — 3-D view of a recovered model next to the true bodies.

Both models are sampled on one regular 20 m grid over the survey volume.
Truth is drawn as its bodies (vein + deep conductor in red, stock + cap in blue);
the inversion as translucent voxels at two thresholds, strict (<40 / >1500
ohm-m) and relaxed (<100 / >1000). Voxels are drawn on a 40 m grid in physical
coordinates (a 2x2x2 block is filled if any true-body cell / the block-mean
log10 resistivity passes). Scores on the 20 m grid: recall = share of each true
body's volume inside the threshold; precision = share of the thresholded volume
that belongs to a true body of that kind, outside the weathered cover (the
cover is 70 ohm-m, so a recovered cover is not counted as a false conductor).

Env: RUN (bench_dctorch_f32_dh10_chi1.npz), DH (10), PNG (bench_view3d_dh10_chi1.png)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch  # noqa: F401  (before simpeg/pydiso: MKL DLL clash)

import synth3d_common as S
import synth3d_bench as Bn

RUN = os.environ.get("RUN", "bench_dctorch_f32_dh10_chi1.npz")
DH = float(os.environ.get("DH", "10"))
PNG = os.environ.get("PNG", "bench_view3d_dh10_chi1.png")

z = np.load(os.path.join(Bn.OUT, RUN))
row = json.loads(str(z["row"]))
sv = S.build_survey()
mesh, active = S.build_mesh(DH, sv, S.topo_points())
full = np.full(mesh.n_cells, np.nan)
full[active] = z["m_log_rho"] / np.log(10)

step = 20.0
gx = np.arange(-1500.0, 1500.0, step) + step / 2          # even counts: 150, 110, 50
gy = np.arange(-1100.0, 1100.0, step) + step / 2
tmax = S.topo_z(*np.meshgrid(gx, gy)).max()
gz = np.arange(tmax - 1000.0, tmax, step) + step / 2
X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
topo = S.topo_z(X, Y)
air = Z > topo
inv = full[mesh.point2index(P)].reshape(X.shape)
inv[air | ~np.isfinite(inv)] = np.log10(S.RHO["host"])
masks = {k: v.reshape(X.shape) & ~air for k, v in S.unit_masks(P).items()}
body = {u: masks[u] & ~masks["cover"] for u in ("vein", "deep", "stock", "cap")}
true_c, true_r = body["vein"] | body["deep"], body["stock"] | body["cap"]
cover = masks["cover"]

LEVELS = [("strict", 40.0, 1500.0), ("relaxed", 100.0, 1000.0)]


def pool(a, how):
    n = [s // 2 * 2 for s in a.shape]
    b = a[:n[0], :n[1], :n[2]].reshape(n[0] // 2, 2, n[1] // 2, 2, n[2] // 2, 2)
    return b.any(axis=(1, 3, 5)) if how == "any" else b.mean(axis=(1, 3, 5))


edges = [np.r_[g[0] - step / 2 + np.arange(len(g) // 2) * 2 * step, g[0] - step / 2 + len(g) // 2 * 2 * step]
         for g in (gx, gy, gz)]
EX, EY, EZ = np.meshgrid(*edges, indexing="ij")
ue = sv.unique_electrode_locations


def draw(ax, fill_c, fill_r, title):
    if fill_c.any():
        ax.voxels(EX, EY, EZ, fill_c, facecolors="#d7301f", edgecolor="none", alpha=0.35)
    if fill_r.any():
        ax.voxels(EX, EY, EZ, fill_r, facecolors="#2166ac", edgecolor="none", alpha=0.30)
    ax.scatter(ue[:, 0], ue[:, 1], ue[:, 2], s=1, c="k", depthshade=False)
    ax.set(xlim=(edges[0][0], edges[0][-1]), ylim=(edges[1][0], edges[1][-1]),
           zlim=(edges[2][0], edges[2][-1]), xlabel="E (m)", ylabel="N (m)", zlabel="elev (m)")
    ax.set_box_aspect((np.ptp(edges[0]), np.ptp(edges[1]), np.ptp(edges[2])))
    ax.view_init(elev=30, azim=-60)
    ax.set_title(title, fontsize=9)


fig = plt.figure(figsize=(21, 7.5))
ax = fig.add_subplot(1, 3, 1, projection="3d")
draw(ax, pool(true_c, "any"), pool(true_r, "any"),
     "truth: vein + deep conductor (red), stock + cap (blue)")
inv_pool = pool(inv, "mean")
report = []
for k, (name, lo, hi) in enumerate(LEVELS, start=2):
    ax = fig.add_subplot(1, 3, k, projection="3d")
    draw(ax, inv_pool <= np.log10(lo), inv_pool >= np.log10(hi),
         f"inverted, this engine dh={DH:g} | chi2 {row['chi2_final']} | {row['t_total_after_mesh_s']/60:.1f} min\n"
         f"{name}: red < {lo:g} ohm-m, blue > {hi:g} ohm-m")
    sel_c, sel_r = inv <= np.log10(lo), inv >= np.log10(hi)
    prec_c = np.sum(sel_c & true_c & ~cover) / max(np.sum(sel_c & ~cover), 1)
    prec_r = np.sum(sel_r & true_r) / max(np.sum(sel_r), 1)
    report.append(f"{name:<8} (<{lo:g} / >{hi:g} ohm-m): recall vein {np.mean(sel_c[body['vein']])*100:3.0f}%, "
                  f"deep {np.mean(sel_c[body['deep']])*100:3.0f}%, stock {np.mean(sel_r[body['stock']])*100:3.0f}%, "
                  f"cap {np.mean(sel_r[body['cap']])*100:3.0f}% | precision conductors (outside cover) "
                  f"{prec_c*100:3.0f}%, resistors {prec_r*100:3.0f}%")
fig.text(0.01, 0.01, "\n".join(report), fontsize=9, family="monospace")
fig.savefig(os.path.join(Bn.OUT, PNG), dpi=110, bbox_inches="tight")
print("\n".join(report))
print(f"[fig] {os.path.join(Bn.OUT, PNG)}")
