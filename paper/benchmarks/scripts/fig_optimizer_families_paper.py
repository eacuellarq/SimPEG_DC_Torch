"""Optimizer comparison on the Century line — paper version.

Rebuild of fig_century_ablation.py with the layout tidied for print: one
shared colour bar clear of the panels, two-line titles that do not clip,
correlations quoted at the precision the data actually has, axis furniture
only where it is needed, and the AdamW panel marked as the finding.

Reads paper/benchmarks/results, writes paper/fig_optimizer_families_public.png
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
import century_common as cc
cc.read_century_obs.__defaults__ = (
    os.path.join(HERE, "..", "data_century", "46800E", "46800POT.OBS"),)
DATA = os.path.join(HERE, "..", "results")
OUT = os.path.normpath(os.path.join(
    HERE, "..", "..", "fig_optimizer_families_public.png"))

INK, MUTE, RULE = "#1a1f28", "#6b7280", "#c8ccd2"
TRAP, ADOPT = "#a33a1f", "#1c4078"

survey, dobs, std = cc.build_survey()
mesh, dx = cc.build_mesh(survey)
xe = survey.unique_electrode_locations[:, 0]
ze = np.zeros_like(xe)

rows = {json.loads(ln)["kind"]: json.loads(ln)
        for ln in open(os.path.join(DATA, "ablation_century.jsonl"))}
norm = LogNorm(vmin=30, vmax=1000)
CMAP = mpl.cm.RdYlBu_r

ORDER = ["lbfgsP", "lbfgsR", "adam01", "adam4x", "adamw01", "sgdclip"]
LABEL = {"lbfgsP": "L-BFGS, persistent  (adopted)",
         "lbfgsR": "L-BFGS, restarted per level",
         "adam01": "Adam",
         "adam4x": "Adam, 4$\\times$ budget",
         "adamw01": "AdamW  —  weight-decay trap",
         "sgdclip": "SGD with gradient clipping"}

fig, axs = plt.subplots(3, 2, figsize=(10.4, 6.4), dpi=200)
fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0.075, right=0.855, top=0.925, bottom=0.075,
                    hspace=0.40, wspace=0.13)

for k, (ax, kind) in enumerate(zip(axs.ravel(), ORDER)):
    r = rows[kind]
    m = np.load(os.path.join(DATA, f"ablation_century_{kind}.npz"))["m_log_rho"]
    mesh.plot_image(np.exp(m), ax=ax, grid=False,
                    pcolor_opts={"cmap": CMAP, "norm": norm})
    ax.plot(xe, ze, marker="v", ls="none", ms=2.6, color="#111418",
            clip_on=False, zorder=6)

    ax.set_xlim(xe.min() - 60, xe.max() + 60)
    ax.set_ylim(-420, 30)
    ax.set_aspect(2.4)                       # vertical exaggeration, in caption
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=7.5, colors=MUTE, length=2.5)
    for sp in ax.spines.values():
        sp.set_color(RULE)

    col = TRAP if kind == "adamw01" else (ADOPT if kind == "lbfgsP" else INK)
    ax.text(0.5, 1.30, LABEL[kind], transform=ax.transAxes, ha="center",
            va="bottom", fontsize=9.2, color=col,
            fontweight="bold" if kind in ("lbfgsP", "adamw01") else "normal")

    miss = "" if r["target_hit"] else "  ·  target not reached"
    corr = ("reference model" if kind == "lbfgsP"
            else f"corr {r['corr_vs_lbfgsP']:.4f}")
    ax.text(0.5, 1.06, f"{r['closures']} evaluations   ·   "
                       f"$\\chi$ {r['chi']:.3f}   ·   {corr}{miss}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.0, color=MUTE)

    if k % 2 == 0:
        ax.set_ylabel("depth (m)", fontsize=8.5, color=INK)
    else:
        ax.set_yticklabels([])
    if k >= 4:
        ax.set_xlabel("northing (m)", fontsize=8.5, color=INK)
    else:
        ax.set_xticklabels([])

cax = fig.add_axes([0.875, 0.24, 0.015, 0.48])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
cb.set_label("resistivity  $\\rho$  ($\\Omega\\,$m)", fontsize=8.8, color=INK)
cb.ax.tick_params(labelsize=8, colors=MUTE, length=2)
cb.outline.set_edgecolor(RULE)

fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print("wrote", os.path.abspath(OUT))
for kind in ORDER:
    r = rows[kind]
    print(f"  {kind:9s} closures {r['closures']:6d}  chi {r['chi']:.3f}"
          f"  corr {r['corr_vs_lbfgsP']:.4f}  target_hit {r['target_hit']}")
