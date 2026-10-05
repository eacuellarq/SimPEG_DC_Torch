"""
plot_century_ip.py — Century 46800E: secciones de cargabilidad recuperada
(vanilla GN stock / dctorch TorchIP2D F64) con la seccion geologica de
Leapfrog superpuesta, + convergencia chi vs tiempo (convencion Σ; dctorch
hist lleva ½ -> x2). Mismo estilo que plot_century_models. Lee/escribe en
paper/benchmarks. CWD scripts/, env simpeg311.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

from century_common import REPO
from century_ip_common import DATA as DATA_IP, build_dc_survey_and_mesh

DATA = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e7e4"
BLUE, GREEN, GRAY = "#2a78d6", "#008300", "#52514e"

survey, mesh, dx = build_dc_survey_and_mesh()
nD = 151.0


def load_geology():
    fid = open(os.path.join(os.path.dirname(DATA_IP), "geologic_section.csv"))
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
van = np.load(os.path.join(DATA, "inv_century_ip_vanilla.npz"))
f64 = np.load(os.path.join(DATA, "inv_century_ip_dctorch_f64.npz"))
f32 = np.load(os.path.join(DATA, "inv_century_ip_dctorch_f32.npz"))
nD = float(van["nD"])

models = {
    f"SimPEG IP GN + Pardiso, CPU ({float(van['t']):.1f} s, "
    f"$\\chi^2$ {float(van['chi']):.2f})": van["eta"],
    f"dctorch TorchIP2D F64, GPU ({float(f64['t']) + float(f64['t_build']):.1f} s, "
    f"$\\chi^2$ {float(f64['chi']):.2f})": f64["eta"],
    f"dctorch TorchIP2D F32, GPU ({float(f32['t']) + float(f32['t_build']):.1f} s, "
    f"$\\chi^2$ {float(f32['chi']):.2f})": f32["eta"],
}
xe = survey.unique_electrode_locations[:, 0]
vmax = float(np.percentile(np.r_[van["eta"], f64["eta"]], 99.5))
norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

fig, axs = plt.subplots(3, 1, figsize=(9.5, 10.2))
fig.subplots_adjust(hspace=0.5)
for ax, (name, e) in zip(axs, models.items()):
    ax.set_rasterization_zorder(1)
    mesh.plot_image(e, ax=ax, pcolor_opts={
        "cmap": mpl.cm.plasma, "norm": norm})
    for g in geo:
        ax.plot(g[:, 0], g[:, 1], "w-", lw=1.0, alpha=0.8)
    ax.plot(xe, np.zeros_like(xe), "kv", ms=3)
    ax.set_xlim([xe.min() - 100, xe.max() + 100])
    ax.set_ylim([-500, 30])
    ax.set_title(name, fontsize=12, pad=12)
    ax.set_xlabel("Northing (m)")
    ax.set_ylabel("z (m)")
sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma)
fig.colorbar(sm, ax=axs, shrink=0.65, pad=0.02,
             label=r"$\eta$ (mV/V)")
out = os.path.join(OUT, "fig_century_ip_models.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")
corr = np.corrcoef(van["eta"], f64["eta"])[0, 1]
print(f"corr(eta_vanilla, eta_dctorch) = {corr:.4f}")

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
series = [("SimPEG IP GN (CPU)", L[:, 1], L[:, 2] / nD, GRAY, "^"),
          ("dctorch TorchIP2D F64 (GPU)", np.cumsum(f64["hist"][:, 3]),
           2 * f64["hist"][:, 0] / nD, GREEN, "s"),
          ("dctorch TorchIP2D F32 (GPU)", np.cumsum(f32["hist"][:, 3]),
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
ax.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper right", handlelength=1.6)
for row, (name, t, chi, col, mk) in enumerate(series):
    ax.annotate(f"{t[-1]:.1f} s", xy=(len(chi), 1.03 + 0.055 * (2 - row)),
                xycoords=("data", "axes fraction"), ha="center",
                fontsize=6.3, color=col)
ax.annotate("wall-clock:", xy=(0.0, 1.03 + 0.055 * 3),
            xycoords="axes fraction", fontsize=6.3, color=INK2,
            style="italic")
fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.96))
out = os.path.join(OUT, "fig_century_ip_convergence.png")
fig.savefig(out, bbox_inches="tight")
print(f"saved {out}")
for name, t, chi, *_ in series:
    print(f"  {name}: {len(chi)} iters | t {t[-1]:.1f}s | chi {chi[-1]:.3f}")
