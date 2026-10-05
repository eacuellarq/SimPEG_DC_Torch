"""Optimizer comparison figure, built from the native ablation results.

Panel (a): cost (closures) against final data fit, with the discrepancy
target marked.  Panel (b): the models recovered by the adopted optimizer
and by AdamW, which reaches an apparently healthy fit on a different model.

Data: results/ablation_century.jsonl and ablation_century_*.npz
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "..", "fig_optimizers_public.png")

INK, MUTE, RULE = "#1a1f28", "#6b7280", "#c8ccd2"
ADOPT, TRAP, PLAIN, TGT = "#1c4078", "#a33a1f", "#9aa4b2", "#8a4a2b"

LABEL = {"lbfgsP": "quasi-Newton\n(adopted)",
         "lbfgsR": "quasi-Newton,\nrestarted",
         "adam01": "Adam",
         "adam4x": "Adam, 4× budget",
         "adamw01": "AdamW",
         "sgdclip": "gradient descent"}

runs = {}
with open(os.path.join(RES, "ablation_century.jsonl")) as fh:
    for line in fh:
        d = json.loads(line)
        runs[d["kind"]] = d

fig, (ax, ax2) = plt.subplots(
    1, 2, figsize=(10.6, 3.5), dpi=200,
    gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.28})
fig.patch.set_facecolor("white")

# ------------------------------------------------- (a) cost against the fit
ax.set_facecolor("white")
ax.axhline(1.0, color=TGT, lw=1.4, ls=(0, (5, 3)), zorder=2)
ax.text(78, 1.09, "discrepancy target", color=TGT, fontsize=8.5,
        ha="left", va="bottom")

for kind, d in runs.items():
    col = ADOPT if kind == "lbfgsP" else (TRAP if kind == "adamw01" else PLAIN)
    big = kind in ("lbfgsP", "adamw01")
    ax.scatter(d["closures"], d["chi"], s=95 if big else 55, color=col,
               zorder=5, edgecolor="white", linewidth=1.2)

OFFS = {"lbfgsR":  (-6, -16, "right", "top"),
        "lbfgsP":  (2, 15, "left", "bottom"),
        "adam01":  (0, -16, "center", "top"),
        "adam4x":  (0, 15, "center", "bottom"),
        "adamw01": (12, 0, "left", "center"),
        "sgdclip": (-8, -6, "right", "top")}
for kind, (ox, oy, ha, va) in OFFS.items():
    d = runs[kind]
    col = ADOPT if kind == "lbfgsP" else (TRAP if kind == "adamw01" else MUTE)
    ax.annotate(LABEL[kind], xy=(d["closures"], d["chi"]),
                xytext=(ox, oy), textcoords="offset points",
                ha=ha, va=va, fontsize=8.5, color=col, linespacing=1.25)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(75, 4.0e4)
ax.set_ylim(0.16, 16)
ax.set_xlabel("objective evaluations (closures)", fontsize=9.5, color=INK)
ax.set_ylabel("final data fit  $\\chi$", fontsize=9.5, color=INK)
ax.tick_params(labelsize=8.5, colors=MUTE, length=3)
ax.grid(True, which="major", color=RULE, lw=0.6, alpha=0.7, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(RULE)
ax.text(0.0, 1.04, "(a)  cost against the fit achieved",
        transform=ax.transAxes, fontsize=10, color=INK, va="bottom")

# ------------------------------------- (b) the models these choices produce
import century_common as cc
# build_survey() binds the OBS default at import time, so patch the default
_OBS = os.path.join(HERE, "..", "data_century", "46800E", "46800POT.OBS")
cc.read_century_obs.__defaults__ = (_OBS,)

survey, dobs, std = cc.build_survey()
mesh, dx = cc.build_mesh(survey)

mods = {k: np.load(os.path.join(RES, f"ablation_century_{k}.npz"))["m_log_rho"]
        for k in ("lbfgsP", "adamw01")}
rho = {k: np.exp(v) for k, v in mods.items()}
vmin = min(r.min() for r in rho.values())
vmax = max(r.max() for r in rho.values())

x0, x1 = survey.unique_electrode_locations[:, 0].min(), \
    survey.unique_electrode_locations[:, 0].max()
zmin = -360.0

ax2.set_facecolor("white")
gs = ax2.get_subplotspec().subgridspec(2, 1, hspace=0.16)
ax2.remove()
axes = [fig.add_subplot(gs[i]) for i in range(2)]

for a, kind, title in zip(axes, ("lbfgsP", "adamw01"),
                          ("quasi-Newton (adopted)",
                           "AdamW  —  $\\chi=0.28$, model correlation 0.98")):
    im = mesh.plot_image(np.log10(rho[kind]), ax=a, grid=False,
                         pcolor_opts={"cmap": "Spectral_r",
                                      "vmin": np.log10(vmin),
                                      "vmax": np.log10(vmax)})[0]
    a.set_xlim(x0 - 40, x1 + 40)
    a.set_ylim(zmin, 20)
    a.set_aspect(3.0)      # 3x vertical exaggeration, stated in the caption
    a.set_title("")
    a.set_xlabel("")
    a.tick_params(labelsize=8, colors=MUTE, length=3)
    a.set_ylabel("depth (m)", fontsize=9, color=INK)
    col = ADOPT if kind == "lbfgsP" else TRAP
    a.text(0.012, 0.94, title, transform=a.transAxes, fontsize=8.5,
           color=col, va="top",
           bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.8))
    for sp in a.spines.values():
        sp.set_color(RULE)

axes[0].set_xticklabels([])
axes[1].set_xlabel("distance along line (m)", fontsize=9, color=INK)
axes[0].text(0.0, 1.16, "(b)  the models those choices produce",
             transform=axes[0].transAxes, fontsize=10, color=INK, va="bottom")

cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.015, aspect=26)
cb.set_label("$\\log_{10}\\,\\rho$  ($\\Omega$m)", fontsize=8.5, color=INK)
cb.ax.tick_params(labelsize=8, colors=MUTE, length=2)
cb.outline.set_edgecolor(RULE)

fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print("wrote", os.path.abspath(OUT))
