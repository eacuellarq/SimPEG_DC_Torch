"""
fig_public_scaling.py — versiones 100% REPRODUCIBLES (problema dos-esferas,
696 datos publicos) de las tres figuras de escalado/memoria:
  fig_eval_cost_public.png | fig_speedup_public.png | fig_memory_wall_public.png
Mismo estilo/paleta que las versiones del survey privado. Lee de
paper/benchmarks/results (bench_*_spheres.jsonl, gn_storej_ram_spheres.json),
escribe en paper/benchmarks/figures.
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
BLUE, GREEN, GRAY = "#2a78d6", "#008300", "#52514e"

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

van = sorted((json.loads(ln) for ln in open(os.path.join(
    DATA, "bench_vanilla3d_spheres.jsonl")) if "killed" not in ln),
    key=lambda r: r["n_active"])
dct = [json.loads(ln) for ln in open(os.path.join(
    DATA, "bench_scale3d_spheres.jsonl")) if "killed" not in ln]
f32 = sorted((r for r in dct if r["prec"] == "f32"),
             key=lambda r: r["n_active"])
f64 = sorted((r for r in dct if r["prec"] == "f64"),
             key=lambda r: r["n_active"])

nv = np.array([r["n_active"] for r in van], float)
tv = np.array([r["t_fwd"] + r["t_grad"] for r in van])
n32 = np.array([r["n_active"] for r in f32], float)
t32 = np.array([r["t_grad_total"] for r in f32])
n64 = np.array([r["n_active"] for r in f64], float)
t64 = np.array([r["t_grad_total"] for r in f64])

# ---------------- eval cost ----------------
fig, ax = plt.subplots(figsize=(3.5, 2.9))
ax.loglog(nv, tv, "-", color=GRAY, lw=1.3, marker="^", ms=3.5, mfc="white",
          mew=0.9, mec=GRAY, dashes=(4, 1.6))
ax.loglog(n64, t64, "-", color=GREEN, lw=1.3, marker="s", ms=3, mfc="white",
          mew=0.9, mec=GREEN)
ax.loglog(n32, t32, "-", color=BLUE, lw=1.3, marker="o", ms=3.2, mfc="white",
          mew=0.9, mec=BLUE)
ax.text(nv[-1] * 1.12, tv[-1], "SimPEG + Pardiso\n(CPU)", color=GRAY,
        fontsize=7, va="center")
ax.text(n64[-1] * 1.12, t64[-1], "This engine, F64", color=GREEN, fontsize=7,
        va="center")
ax.text(n32[-1] * 1.12, t32[-1] * 0.88, "This engine, F32", color=BLUE,
        fontsize=7, va="center")
i = int(np.argmin(np.abs(n32 - nv[-1])))
sp = tv[-1] / t32[i]
ax.annotate("", xy=(nv[-1], t32[i]), xytext=(nv[-1], tv[-1]),
            arrowprops=dict(arrowstyle="|-|,widthA=0.15,widthB=0.15",
                            color=INK2, lw=0.7, linestyle=(0, (2, 2))))
ax.text(nv[-1] * 0.88, np.sqrt(tv[-1] * t32[i]), f"{sp:,.0f}×", color=INK,
        fontsize=8.5, ha="right", va="center", fontweight="bold")
ax.set_xlabel("Model parameters $N$")
ax.set_ylabel("Misfit + gradient evaluation (s)")
ax.set_xticks([2e4, 5e4, 1e5, 2e5],
              ["2×10$^4$", "5×10$^4$", "10$^5$", "2×10$^5$"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(axis="y", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, "fig_eval_cost_public.png"),
            bbox_inches="tight")
print(f"saved fig_eval_cost_public.png (gap dh comun: {sp:.0f}x)")

# ---------------- speedup bars ----------------
d32 = {r["dh"]: r for r in f32}
d64 = {r["dh"]: r for r in f64}
common = [r for r in van if r["dh"] in d32 and r["dh"] in d64]
nn = np.array([r["n_active"] for r in common], float)
s32 = np.array([(r["t_fwd"] + r["t_grad"]) / d32[r["dh"]]["t_grad_total"]
                for r in common])
s64 = np.array([(r["t_fwd"] + r["t_grad"]) / d64[r["dh"]]["t_grad_total"]
                for r in common])
fig, ax = plt.subplots(figsize=(3.5, 2.9))
x = np.arange(len(common), dtype=float)
w = 0.38
b32 = ax.bar(x - w/2, s32, w * 0.94, color=BLUE, zorder=3)
b64 = ax.bar(x + w/2, s64, w * 0.94, color=GREEN, zorder=3)
ymax = max(s32.max(), s64.max())
for xi, v in list(zip(x - w/2, s32)) + list(zip(x + w/2, s64)):
    ax.text(xi, v + ymax * 0.02, f"{v:.0f}", ha="center", fontsize=6.5,
            color=INK)
ax.axhline(1, color=INK2, lw=0.7)
ax.legend([b32, b64], ["This engine, F32", "This engine, F64"], loc="upper left",
          handlelength=1.2, handleheight=1.1, borderaxespad=0.2)
ax.set_xlabel("Model parameters $N$ (×10³)")
ax.set_ylabel("Speed-up vs SimPEG + Pardiso, CPU (×)")
ax.set_ylim(0, ymax * 1.15)
ax.set_xticks(x, [f"{n/1e3:.0f}" for n in nn])
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", length=0)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, "fig_speedup_public.png"),
            bbox_inches="tight")
print("saved fig_speedup_public.png")
print("speedups F32:", dict(zip([r["dh"] for r in common], np.round(s32))))

# ---------------- memory (lineal, solo datos) ----------------
gn = json.load(open(os.path.join(DATA, "gn_storej_ram_spheres.json")))
gn_n = np.array([r["n_active"] for r in gn], float)
gn_m = np.array([r["ram_gib"] for r in gn])
fig, ax = plt.subplots(figsize=(4.4, 3.2))
a, b = np.polyfit(gn_n, gn_m, 1)
nf = np.linspace(gn_n.min(), gn_n.max(), 10)
ax.plot(nf / 1e5, a * nf + b, "-", color=INK, lw=1.2, zorder=2,
        label="SimPEG GN storeJ, measured RAM (fit)")
ax.plot(gn_n / 1e5, gn_m, "^", color=GRAY, ms=6, mfc="white", mew=1.1,
        zorder=3)
ax.text(gn_n[-1] / 1e5 + 0.07, gn_m[-1] - 0.12,
        rf"$\mathcal{{O}}(N \cdot N_d)$" + "\n"
        rf"$\approx$ {a*2**20:.0f} KiB / parameter",
        color=INK, fontsize=6.8, va="top")
for prec, col, mk, lab, d in [
        ("f64", GREEN, "s", "This engine, F64, total GPU state", f64),
        ("f32", BLUE, "o", "This engine, F32, total GPU state", f32)]:
    n = np.array([r["n_active"] for r in d], float)
    mm = np.array([r["nvml_peak_gib"] for r in d])
    ax.plot(n / 1e5, mm, "-", color=col, lw=1.3, marker=mk, ms=3.2,
            mfc="white", mew=0.8, mec=col, label=lab)
    ad, bd = np.polyfit(n, mm, 1)
    ax.text(n[-1] / 1e5 + 0.06, mm[-1],
            rf"$\mathcal{{O}}(N)$" + "\n"
            rf"$\approx$ {ad*2**20:.0f} KiB / parameter",
            color=col, fontsize=6.8, va="center")
ax.set_xlabel("Model parameters $N$ (×10$^5$)")
ax.set_ylabel("Peak memory (GiB)")
ax.grid(True, which="major", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", handlelength=1.5)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, "fig_memory_wall_public.png"),
            bbox_inches="tight")
print("saved fig_memory_wall_public.png | GN slope "
      f"{a*2**20:.0f} KiB/param (nD=696)")
