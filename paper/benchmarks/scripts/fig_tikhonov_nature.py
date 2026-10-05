"""
fig_tikhonov_nature.py — dos figuras estilo Nature del benchmark dos-esferas
dh=25 (receta del tutorial oficial):

  fig_tikhonov.png    — curvas de Tikhonov clasicas: iteracion en X,
                        phi_d (Y izq, solido) y phi_m (Y der, punteado).
  fig_convergence.png — chi² en Y compartido; dos paneles: vs iteracion
                        y vs wall-clock (ticks planos 1/10/100/1000).

Convencion Σ (sin ½): dctorch (½) x2. Fuentes: iter_log del vanilla
cronometrado + hist de los npz dctorch.

Genera DOS variantes: con sensitivity weights (sufijo vacio, dctorch npz sin
tag) y sin ellos (sufijo _nosw, npz _nosw) — el vanilla (que siempre lleva
pesos, es su flujo) es el mismo en ambas.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), r"paper\benchmarks"))
DATA = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e7e4"
BLUE, GREEN, GRAY = "#2a78d6", "#008300", "#52514e"
ND = 696.0

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.size": 0, "ytick.minor.size": 0,
    "legend.frameon": False, "legend.fontsize": 7,
    "savefig.dpi": 400, "savefig.facecolor": "white",
})

def load_series(tag):
    series = []
    van = np.load(os.path.join(DATA, "inv_tutorial3d_vanilla_dh25.npz"))
    L = van["iter_log"]                     # (iter, t, phi_d, phi_m, beta)
    series.append(("SimPEG GN (CPU)", L[:, 1], L[:, 2], L[:, 3], GRAY, "^"))
    for prec, col, mk in [("f64", GREEN, "s"), ("f32", BLUE, "o")]:
        d = np.load(os.path.join(
            DATA, f"inv_tutorial3d_dctorch_{prec}_dh25{tag}.npz"))
        h = d["hist"]                       # (pd½, pm½, beta, t_outer)
        series.append((f"dctorch {prec.upper()} (GPU)", np.cumsum(h[:, 3]),
                       2 * h[:, 0], 2 * h[:, 1], col, mk))
    return series

MK = dict(ms=3.0, mfc="white", mew=0.8)


def style_ax(ax):
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)



for tag in ("", "_nosw"):
    series = load_series(tag)
    print(f"=== variante {tag or 'sensw'} ===")
    # ================= Fig 1: curvas de Tikhonov (doble Y) =================
    n_all = max(len(s[2]) for s in series)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    axr = ax.twinx()
    for name, t, pd_, pm_, col, mk in series:
        it = np.arange(1, len(pd_) + 1)
        ax.semilogy(it, pd_, "-", color=col, lw=1.3, marker=mk, mec=col,
                    label=name, **MK)
        axr.semilogy(it, pm_, "--", color=col, lw=0.9, alpha=0.55)
    ax.axhline(ND, color=INK2, lw=0.6, dashes=(2, 2))
    ax.text(1.1, ND * 1.12, "target", color=INK2, fontsize=6.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Data misfit $\phi_d$  (solid)")
    axr.set_ylabel(r"Model norm $\phi_m$  (dashed)", color=INK2)
    ax.set_xticks(np.arange(1, n_all + 1, 2))
    ax.set_yticks([700, 1000, 2000, 5000],
                  ["700", "1000", "2000", "5000"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axr.tick_params(colors=INK2)
    axr.spines["top"].set_visible(False)
    style_ax(ax)
    axr.grid(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncols=3,
              handlelength=1.3, columnspacing=1.0, borderaxespad=0.0)
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.94))
    out1 = os.path.join(OUT, f"fig_tikhonov{tag}.png")
    fig.savefig(out1, bbox_inches="tight")
    print(f"saved {out1}")

    # ==== Fig 2: chi² vs iteracion, con split-times de wall-clock arriba ====
    fig2, ax = plt.subplots(figsize=(4.6, 3.3))
    n_it_max = max(len(s[2]) for s in series)
    for name, t, pd_, pm_, col, mk in series:
        it = np.arange(1, len(pd_) + 1)
        ax.semilogy(it, pd_ / ND, "-", color=col, lw=1.3, marker=mk, mec=col,
                    label=name, **MK)
    ax.axhline(1.0, color=INK2, lw=0.6, dashes=(2, 2))
    ax.text(1.2, 1.07, "target", color=INK2, fontsize=6.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\chi^2 = \phi_d / N_d$")
    ax.set_xticks(np.arange(1, n_it_max + 1, 2))
    ax.set_yticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style_ax(ax)
    ax.legend(loc="upper right", handlelength=1.6)

    # split-times: tiempo acumulado en iteraciones comunes + el TOTAL de cada
    # motor en SU iteracion final (lo que le costo converger, iters de mas
    # incluidas)
    n_common = min(len(s[1]) for s in series)
    marks_common = [i for i in (1, 5, 9, 13) if i < n_common]
    for row, (name, t, pd_, pm_, col, mk) in enumerate(series):
        y = 1.03 + 0.055 * (len(series) - 1 - row)
        for i in marks_common + [len(t)]:
            ax.annotate(f"{t[i-1]:,.0f} s", xy=(i, y),
                        xycoords=("data", "axes fraction"), ha="center",
                        fontsize=6.3, color=col)
    ax.annotate("wall-clock:", xy=(0.0, 1.03 + 0.055 * len(series)),
                xycoords="axes fraction", fontsize=6.3, color=INK2,
                style="italic")
    fig2.tight_layout(pad=0.5, rect=(0, 0, 1, 0.97))
    out2 = os.path.join(OUT, f"fig_convergence{tag}.png")
    fig2.savefig(out2, bbox_inches="tight")
    print(f"saved {out2}")

    for name, t, pd_, pm_, col, mk in series:
        print(f"  {name}: {len(pd_)} iters | t {t[-1]:.1f}s | "
              f"chi {pd_[-1]/ND:.3f}")
