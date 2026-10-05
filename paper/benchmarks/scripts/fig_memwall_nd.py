"""
fig_memwall_nd.py — LA figura de la pared de memoria (publica y dramatica
sin proyecciones): pico de memoria vs numero de datos N_d a malla fija
(dos-esferas dh=25, survey dipolo-dipolo densificado sobre el MISMO modelo
verdadero). GN storeJ crece ~linealmente con N_d (la J explicita es N·N_d);
el estado GPU completo de dctorch es esencialmente PLANO en N_d.
Lee results/memwall_vs_nd.jsonl -> figures/fig_memory_wall_public.png.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), r"paper\benchmarks"))
DATA = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e7e4"
BLUE, GRAY = "#2a78d6", "#52514e"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.size": 0, "ytick.minor.size": 0,
    "legend.frameon": False, "legend.fontsize": 6.8,
    "savefig.dpi": 400, "savefig.facecolor": "white",
})

rows = [json.loads(l) for l in open(os.path.join(DATA,
                                                 "memwall_vs_nd.jsonl"))]
gn = sorted((r for r in rows if r["engine"].startswith("gn")),
            key=lambda r: r["nD"])
dc = sorted((r for r in rows if r["engine"].startswith("dctorch")),
            key=lambda r: r["nD"])

fig, ax = plt.subplots(figsize=(4.4, 3.2))
ngn = np.array([r["nD"] for r in gn], float)
mgn = np.array([r["mem_gib"] for r in gn])
ndc = np.array([r["nD"] for r in dc], float)
mdc = np.array([r["mem_gib"] for r in dc])

a, b = np.polyfit(ngn, mgn, 1)
nf = np.linspace(ngn.min(), ngn.max(), 10)
ax.plot(nf / 1e3, a * nf + b, "-", color=INK, lw=1.2, zorder=2)
ax.plot(ngn / 1e3, mgn, "^", color=GRAY, ms=6, mfc="white", mew=1.1,
        zorder=3, label="SimPEG GN storeJ (CPU RAM)")
ax.plot(ndc / 1e3, mdc, "o-", color=BLUE, lw=1.3, ms=3.6, mfc="white",
        mew=0.9, mec=BLUE, zorder=3,
        label="dctorch F32 (total GPU state)")

ax.text(ngn[-1] / 1e3 - 0.35, mgn[-1] + 0.30,
        rf"$\mathcal{{O}}(N \cdot N_d)$" + "\n"
        + rf"$\approx$ {a*2**30/2**20*1e3:,.0f} MiB / 1,000 data",
        color=INK, fontsize=6.8, ha="left", va="bottom")
adc = np.polyfit(ndc, mdc, 1)[0]
ax.text(10.5, 3.05,
        rf"$\mathcal{{O}}(N)$" + "\n"
        + rf"$\approx$ {adc*2**30/2**20*1e3:,.0f} MiB / 1,000 data"
        + "\n(stored fields, shared by any adjoint)",
        color=BLUE, fontsize=6.8, ha="left", va="top")
# el otro golpe MEDIDO: el tiempo de 2 iteraciones GN por punto
for x, y, t in zip(ngn / 1e3, mgn, [r["t"] for r in gn]):
    ax.annotate(f"{t/60:.0f} min", xy=(x, y), xytext=(x + 0.30, y - 0.42),
                fontsize=6.2, color=GRAY)
ax.annotate("(wall-clock of\n2 GN iterations)", xy=(1.35, 0.72),
            xycoords="data", fontsize=6.2, color=GRAY, style="italic",
            va="top")

ax.set_xlabel("Number of data $N_d$ (×10³), same mesh")
ax.set_ylabel("Peak memory (GiB)")
ax.grid(True, which="major", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_ylim(0, max(mgn.max(), mdc.max()) * 1.18)
ax.legend(loc="upper left", handlelength=1.5)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, "fig_memory_wall_public.png"),
            bbox_inches="tight")
print(f"saved fig_memory_wall_public.png | GN "
      f"{a*2**30/2**20*1e3:.0f} MiB/1000 datos vs dctorch "
      f"{adc*2**30/2**20*1e3:.0f}")
for r in gn + dc:
    print(f"  {r['engine']:12s} nD {r['nD']:6d} nC {r['nC']:7d} "
          f"{r['mem_gib']:6.2f} GiB  ({r['t']:.0f}s)")
