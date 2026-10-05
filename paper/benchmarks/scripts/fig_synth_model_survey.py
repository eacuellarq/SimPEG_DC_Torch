"""fig_synth_model_survey.py — the field-scale synthetic: survey layout and true model.

(a) Plan view: topography (contour interval 50 m), the electrodes of the eleven
E-W lines and three N-S tie lines, the horizontal footprints of the conductive
and resistive bodies (projected over all depths), and the traces of the two
sections. (b, c) True resistivity along section 1 (deep conductor, vein, stock)
and section 2 (across the vein, down its dip), without vertical exaggeration;
electrodes within 30 m of each trace are projected onto the surface.

Outputs: ../figures/fig_synth_model_survey.{png,pdf}
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import synth3d_common as S

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
INK, MUTE, RULE, TOPO = "#1a1f28", "#6b7280", "#c8ccd2", "#9aa0a6"
COND, RES = "#b2182b", "#2166ac"
NORM = TwoSlopeNorm(vmin=np.log10(5), vcenter=np.log10(S.RHO["host"]), vmax=np.log10(5000))
CMAP = plt.get_cmap("RdBu")
CB_TICKS = [5, 10, 50, 100, 500, 1000, 5000]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})


def style(ax):
    ax.tick_params(colors=MUTE, length=2.5, width=0.6)
    for sp in ax.spines.values():
        sp.set_color(RULE)
        sp.set_linewidth(0.6)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def electrodes():
    xy = [(x, y) for y in S.LINE_Y for x in S.LINE_X] + [(x, y) for x in S.TIE_X for y in S.TIE_Y]
    return np.unique(np.array(xy), axis=0)


def section_defs():
    p_deep, p_stock = np.array(S.DEEP["c"]), np.array(S.STOCK["c"])
    u = (p_stock - p_deep) / np.linalg.norm(p_stock - p_deep)
    half1 = 0.5 * np.linalg.norm(p_stock - p_deep) + 400.0
    _, dd, _ = S._vein_frame()
    dd = dd[:2] / np.linalg.norm(dd[:2])
    return [(0.5 * (p_deep + p_stock), u, half1), (np.array(S.VEIN["p0"]), dd, 1300.0)]


def section_image(c0, d, half, step=5.0, depth_below=650.0):
    s = np.arange(-half, half + step / 2, step)
    xs, ys = c0[0] + s * d[0], c0[1] + s * d[1]
    zt = S.topo_z(xs, ys)
    z = np.arange(zt.min() - depth_below, zt.max() + step, step)
    SG, ZG = np.meshgrid(s, z)
    P = np.c_[np.broadcast_to(xs, SG.shape).ravel(), np.broadcast_to(ys, SG.shape).ravel(), ZG.ravel()]
    img = (S.log_rho(P) / np.log(10)).reshape(SG.shape)
    return s, z, zt, np.ma.masked_where(ZG > zt[None, :], img)


E = electrodes()
SECS = section_defs()

# ---------------- plan-view footprints of the bodies ----------------
gx = np.arange(-1700.0, 1700.0 + 1, 10.0)
gy = np.arange(-1300.0, 1300.0 + 1, 10.0)
X, Y = np.meshgrid(gx, gy)
ZT = S.topo_z(X, Y)
cond = np.zeros(X.shape, bool)
res = np.zeros(X.shape, bool)
for depth in np.arange(0.0, 701.0, 20.0):
    m = S.unit_masks(np.c_[X.ravel(), Y.ravel(), (ZT - depth).ravel()])
    cond |= (m["vein"] | m["deep"]).reshape(X.shape)
    res |= (m["stock"] | m["cap"]).reshape(X.shape)

fig = plt.figure(figsize=(7.2, 4.2), dpi=300)
fig.patch.set_facecolor("white")
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.3], height_ratios=[1, 1], wspace=0.36, hspace=0.55)
ax_a = fig.add_subplot(gs[:, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])

ax = ax_a
ax.contour(X, Y, ZT, levels=np.arange(1300, 1850, 50), colors=TOPO, linewidths=0.35)
ax.contourf(X, Y, cond.astype(float), levels=[0.5, 1.5], colors=[COND], alpha=0.30)
ax.contour(X, Y, cond.astype(float), levels=[0.5], colors=[COND], linewidths=0.7)
ax.contourf(X, Y, res.astype(float), levels=[0.5, 1.5], colors=[RES], alpha=0.30)
ax.contour(X, Y, res.astype(float), levels=[0.5], colors=[RES], linewidths=0.7)
ax.plot(E[:, 0], E[:, 1], "o", ms=1.4, mew=0, color=INK, zorder=4)
for (c0, d, half), ls in zip(SECS, ("-", (0, (5, 2)))):
    p0, p1 = c0 - half * d, c0 + half * d
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=1.3, ls=ls, zorder=5)
ax.set_xlim(gx[0], gx[-1])
ax.set_ylim(gy[0], gy[-1])
ax.set_aspect("equal")
ax.set_xlabel("easting (m)")
ax.set_ylabel("northing (m)")
ax.set_title("(a) Survey and bodies", loc="left", fontweight="bold")
ax.set_anchor("N")
style(ax)
handles = [Line2D([], [], ls="none", marker="o", ms=2.5, mew=0, color=INK, label="electrodes"),
           Line2D([], [], color=TOPO, lw=0.6, label="topography, 50 m contours"),
           Patch(facecolor=COND, alpha=0.30, edgecolor=COND, lw=0.7, label="conductive bodies"),
           Patch(facecolor=RES, alpha=0.30, edgecolor=RES, lw=0.7, label="resistive bodies"),
           Line2D([], [], color=INK, lw=1.3, ls="-", label="section 1"),
           Line2D([], [], color=INK, lw=1.3, ls=(0, (5, 2)), label="section 2")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.19), ncol=1,
          frameon=False, fontsize=6.8, handlelength=2.0, handletextpad=0.6, labelspacing=0.45)

mappable = None
for ax, (c0, d, half), title in ((ax_b, SECS[0], "(b) Section 1"), (ax_c, SECS[1], "(c) Section 2")):
    s, z, zt, img = section_image(c0, d, half)
    mappable = ax.pcolormesh(s, z, img, cmap=CMAP, norm=NORM, shading="nearest", rasterized=True)
    ax.plot(s, zt, color=INK, lw=0.7)
    rel = E - c0
    t = rel @ d
    perp = np.linalg.norm(rel - np.outer(t, d), axis=1)
    k = (perp <= 30.0) & (np.abs(t) <= half)
    ax.plot(t[k], S.topo_z(E[k, 0], E[k, 1]) + 12.0, "v", ms=2.6, mew=0, color=INK, zorder=4)
    ax.set_xlim(-half, half)
    ax.set_ylim(z[0], zt.max() + 40.0)
    ax.set_aspect("equal")
    ax.set_xlabel("distance along section (m)")
    ax.set_ylabel("elevation (m)")
    ax.set_title(title, loc="left", fontweight="bold")
    style(ax)

cb = fig.colorbar(mappable, ax=[ax_b, ax_c], orientation="horizontal", fraction=0.045, pad=0.16, aspect=40)
cb.set_ticks(np.log10(CB_TICKS))
cb.set_ticklabels([str(v) for v in CB_TICKS])
cb.set_label("resistivity (Ω·m)", color=INK)
cb.ax.tick_params(colors=MUTE, length=2.5, width=0.6)
cb.outline.set_edgecolor(RULE)
cb.outline.set_linewidth(0.6)

os.makedirs(FIG, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig_synth_model_survey.{ext}"), facecolor="white", bbox_inches="tight")
print("saved", os.path.join(FIG, "fig_synth_model_survey.png"))
