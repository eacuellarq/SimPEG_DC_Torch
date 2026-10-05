"""The four beta rules, as sections and as convergence.

Reads the .npz that `hybrid_beta_2d.py SAVE=...` writes for its first seed.

The colour map is the diverging one the other sections in this repository use,
and it is diverging on purpose rather than by habit: the model is a deviation
from a known 100 ohm-m halfspace, so the quantity has a real polarity and the
neutral step sits on the background.

  python -u fig_hybrid_beta.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from haar_inversion_2d import build                                # noqa: E402

NPZ = os.environ.get("NPZ", os.path.join(HERE, "..", "results",
                                        "hybrid_beta_2d_seed7.npz"))
OUT = os.environ.get("OUT", os.path.join(HERE, "..", "..", "..", "..",
                                        "_gpubench", "fig_hybrid_beta.png"))
INK, MUTE, RULE = "#1a1f28", "#6b7280", "#c8ccd2"
# one hue per rule, fixed order, never cycled
COL = {"gcv": "#1c4078", "discrepancy": "#2e7d5b",
       "cool": "#a33a1f", "fixed": "#9aa4b2"}
NAME = {"gcv": "GCV", "discrepancy": "discrepancy principle",
        "cool": "cooling schedule", "fixed": "fixed $\\beta$"}
ORDER = ["gcv", "discrepancy", "cool", "fixed"]
RHO_BG = 100.0


def main():
    z = np.load(NPZ)
    mesh, survey, m_true, body = build()
    xe = survey.unique_electrode_locations[:, 0]

    fig = plt.figure(figsize=(13.0, 7.2))
    panels = [("true", m_true)] + [(z_ := r, z[f"m_{r}"])[::1] for r in ORDER]
    panels = [("true", m_true)] + [(r, z[f"m_{r}"]) for r in ORDER]
    lo = min(v.min() for _, v in panels) / np.log(10)
    hi = max(v.max() for _, v in panels) / np.log(10)
    norm = TwoSlopeNorm(vmin=min(lo, np.log10(RHO_BG) - 0.05),
                        vcenter=np.log10(RHO_BG),
                        vmax=max(hi, np.log10(RHO_BG) + 0.05))

    # explicit geometry: five sections, a horizontal bar beneath them, two plots
    L, W = 0.052, 0.455
    top, hgt, gap = 0.760, 0.124, 0.022
    im = None
    for i, (key, m) in enumerate(panels):
        ax = fig.add_axes([L, top - i * (hgt + gap), W, hgt])
        im = mesh.plot_image(m / np.log(10), ax=ax, grid=False,
                             pcolor_opts={"cmap": "RdYlBu_r", "norm": norm})[0]
        ax.set_xlim(xe.min() - 4, xe.max() + 4)
        ax.set_ylim(-24, 1.5)
        ax.set_aspect("auto")
        ax.set_title("")
        ax.set_ylabel("")
        if key == "true":
            head, col = "true model", INK
        else:
            chi2, it, nev, corr = z[f"stat_{key}"]
            head = (f"{NAME[key]}    $\\chi^2$ {chi2:.2f}  ·  {int(it)} steps  ·  "
                    f"{int(nev)} grad-equiv  ·  correlation {corr:.3f}")
            col = COL[key]
        ax.text(0.008, 1.10, head, transform=ax.transAxes, fontsize=9,
                color=col, va="bottom")
        ax.tick_params(labelsize=7.5, colors=MUTE, length=3)
        ax.set_yticks([-20, -10, 0])
        if i < len(panels) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("distance along line (m)", fontsize=9, color=INK)
        if i == 2:
            ax.set_ylabel("depth (m)", fontsize=9, color=INK)
        ax.plot(xe, np.zeros_like(xe), "v", ms=2.6, color=INK,
                clip_on=False, zorder=5)
        for sp in ax.spines.values():
            sp.set_color(RULE)

    cax = fig.add_axes([L + 0.10, 0.072, W - 0.20, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("$\\log_{10}\\,\\rho$  ($\\Omega$m)   —  the line marks the "
                 "100 $\\Omega$m background", fontsize=8.2, color=INK)
    cb.ax.tick_params(labelsize=7.5, colors=MUTE, length=2)
    cb.ax.axvline(np.log10(RHO_BG), color=INK, lw=0.9)

    axc = fig.add_axes([0.605, 0.455, 0.365, 0.350])
    axb = fig.add_axes([0.605, 0.120, 0.365, 0.250])
    xmax = 0.0
    for r in ORDER:
        h = z[f"hist_{r}"]
        axc.plot(h[:, 0], h[:, 1], "-o", ms=4.4, lw=2.0, color=COL[r],
                 label=NAME[r])
        axb.semilogy(h[:, 0], h[:, 2], "-o", ms=4.4, lw=2.0, color=COL[r])
        if r != "fixed":
            xmax = max(xmax, h[:, 0].max())
    xlim = (-15, xmax * 1.18)
    axc.axhline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)))
    axc.text(0.99, 1.06, "target $\\chi^2 = 1$", transform=axc.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=8, color=INK)
    axc.set_yscale("log")
    axc.set_ylabel("$\\chi^2$", fontsize=9, color=INK)
    axc.text(0.0, 1.045, "what each rule costs to get there", fontsize=10,
             color=INK, transform=axc.transAxes, va="bottom")
    axc.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    fixed_h = z["hist_fixed"]
    axc.text(0.015, 0.10,
             f"fixed $\\beta$: still $\\chi^2 \\approx$ {fixed_h[-1, 1]:.0f}\n"
             f"after {int(fixed_h[-1, 0])} grad-equiv",
             transform=axc.transAxes, fontsize=8, color=COL["fixed"],
             va="bottom", linespacing=1.35)
    axb.set_ylabel("$\\beta$ chosen", fontsize=9, color=INK)
    axb.set_xlabel("gradient-equivalents spent", fontsize=9, color=INK)
    for a in (axc, axb):
        a.set_xlim(*xlim)
        a.grid(True, color=RULE, lw=0.6, alpha=0.55)
        a.set_axisbelow(True)
        a.tick_params(labelsize=8, colors=MUTE, length=3)
        for sp in a.spines.values():
            sp.set_color(RULE)
    axc.set_xticklabels([])

    fig.text(L, 0.965,
             "Choosing the regularization parameter inside the Newton step, "
             "against cooling it on a schedule",
             fontsize=12, color=INK, va="top")
    fig.text(L, 0.932,
             f"2.5-D synthetic, {mesh.nC:,} cells, {survey.nD} data, 3 % noise, "
             f"seed {int(z['seed'][0])}.  One Newton step costs {survey.nD} "
             f"gradient-equivalents, because J is formed by {survey.nD} Jtvec calls.",
             fontsize=8.6, color=MUTE, va="top")

    fig.savefig(OUT, dpi=170, facecolor="white")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
