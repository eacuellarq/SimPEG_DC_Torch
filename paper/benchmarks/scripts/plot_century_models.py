"""
plot_century_models.py — Century 46800E: secciones de rho recuperada
(vanilla GN / This engine, F64 / F32) con la seccion geologica de Leapfrog
superpuesta (del repo transform-2020), + curvas de convergencia chi vs
iteracion con split-times (mismo estilo Nature del resto).
Lee/escribe en paper/benchmarks. CWD _gpubench, env simpeg311.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

from century_common import REPO, HERE, build_mesh, build_survey

DATA = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e7e4"
BLUE, GREEN, GRAY = "#2a78d6", "#008300", "#52514e"

survey, dobs, std = build_survey()
mesh, dx = build_mesh(survey)
nD = float(survey.nD)


def load_geology():
    fid = open(os.path.join(HERE, "..", "data_century", "geologic_section.csv"))
    lines = fid.readlines()
    data, tmp = [], []
    for line in lines[2:]:
        if "End" in line:
            if tmp:
                data.append(np.vstack(tmp)[:, [0, 2]])
            tmp = []
        else:
            try:
                tmp.append(np.array(line.split(",")[:3], dtype=float))
            except ValueError:
                pass
    return data


geo = load_geology()
van = np.load(os.path.join(DATA, "inv_century2d_vanilla.npz"))
f64 = np.load(os.path.join(DATA, "inv_century2d_dctorch_f64.npz"))
f32 = np.load(os.path.join(DATA, "inv_century2d_dctorch_f32.npz"))

models = {
    f"SimPEG GN + Pardiso, CPU ({float(van['t']):.1f} s)": van["m_log_rho"],
    f"This engine, F64, GPU ({float(f64['t']) + float(f64['t_build']):.1f} s)":
        f64["m_log_rho"],
    f"This engine, F32, GPU ({float(f32['t']) + float(f32['t_build']):.1f} s)":
        f32["m_log_rho"],
}
xe = survey.unique_electrode_locations[:, 0]
norm = LogNorm(vmin=30, vmax=1000)

fig, axs = plt.subplots(3, 1, figsize=(9.5, 10.2))
fig.subplots_adjust(hspace=0.5)
for ax, (name, m) in zip(axs, models.items()):
    im = mesh.plot_image(np.exp(m), ax=ax, pcolor_opts={
        "cmap": mpl.cm.RdYlBu_r, "norm": norm})[0]
    for g in geo:
        ax.plot(g[:, 0], g[:, 1], "k-", lw=1.0, alpha=0.75)
    ax.plot(xe, np.zeros_like(xe), "kv", ms=3)
    ax.set_xlim([xe.min() - 100, xe.max() + 100])
    ax.set_ylim([-500, 30])
    ax.set_title(name, fontsize=12, pad=12)
    ax.set_xlabel("Northing (m)")
    ax.set_ylabel("z (m)")
sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.RdYlBu_r)
fig.colorbar(sm, ax=axs, shrink=0.65, pad=0.02,
             label=r"$\rho$ ($\Omega\cdot$m)")
out = os.path.join(OUT, "fig_century_models.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# ---- convergencia chi vs iteracion (convencion Σ) ----
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.frameon": False,
    "legend.fontsize": 7, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "savefig.dpi": 400, "savefig.facecolor": "white",
})
L = van["iter_log"]
series = [("SimPEG GN (CPU)", L[:, 1], L[:, 2] / nD, GRAY, "^"),
          ("This engine, F64 (GPU)", np.cumsum(f64["hist"][:, 3]),
           2 * f64["hist"][:, 0] / nD, GREEN, "s"),
          ("This engine, F32 (GPU)", np.cumsum(f32["hist"][:, 3]),
           2 * f32["hist"][:, 0] / nD, BLUE, "o")]
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for name, t, chi, col, mk in series:
    it = np.arange(1, len(chi) + 1)
    ax.semilogy(it, chi, "-", color=col, lw=1.3, marker=mk, ms=3.2,
                mfc="white", mew=0.8, mec=col, label=name)
ax.axhline(1.0, color=INK2, lw=0.6, dashes=(2, 2))
ax.text(1.05, 1.1, "target", color=INK2, fontsize=6.5)
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\chi^2 = \phi_d / N_d$")
ax.set_xticks(np.arange(1, 7))
ax.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper right", handlelength=1.6)
# tiempo total de cada motor en su iteracion final
for row, (name, t, chi, col, mk) in enumerate(series):
    ax.annotate(f"{t[-1]:.1f} s", xy=(len(chi), 1.03 + 0.055 * (2 - row)),
                xycoords=("data", "axes fraction"), ha="center",
                fontsize=6.3, color=col)
ax.annotate("wall-clock:", xy=(0.0, 1.03 + 0.055 * 3),
            xycoords="axes fraction", fontsize=6.3, color=INK2,
            style="italic")
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.96))
out = os.path.join(OUT, "fig_century_convergence.png")
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
for name, t, chi, *_ in series:
    print(f"  {name}: {len(chi)} iters | t {t[-1]:.1f}s | chi {chi[-1]:.3f}")
