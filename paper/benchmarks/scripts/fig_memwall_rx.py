"""The memory wall against survey size, with the injections held fixed.

The earlier version of this figure densified the survey by adding lines, so
every new datum arrived with new current injections attached and the ratio of
data to injections stayed at 7.3 across all four points. Plotted against the
number of data it therefore mixed two axes -- and it understated the very
difference it was drawn to show, because our footprint follows the injections.

Here the mesh (80,508 active cells) and the injection positions (342) are held
fixed and only the number of receivers per injection changes, so the data count
rises ninefold with nothing else moving. The conventional footprint rises with
it; ours does not move at all.

The two curves are drawn on different resources -- host memory for the stored
Jacobian, device memory for ours -- which the caption states. The claim is the
slope, which follows from the formulations: a Jacobian has one row per datum,
fields have one column per injection.

Data: results/memwall_rx.jsonl (bench_memwall_rx_worker.py, 2026-09-07).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "results", "memwall_rx.jsonl"))
OUT = os.path.normpath(os.path.join(HERE, "..", "..",
                                    "fig_memwall_rx_public.png"))

INK, MUTE, RULE = "#1a1f28", "#6b7280", "#c8ccd2"
OURS, GN = "#1c4078", "#a33a1f"

rows = [json.loads(l) for l in open(SRC) if l.strip()]
dc = sorted([r for r in rows if r["engine"].startswith("dctorch")],
            key=lambda r: r["nD"])
gn = sorted([r for r in rows if r["engine"].startswith("gn")],
            key=lambda r: r["nD"])
nC = dc[0]["nC"]
d_nd = np.array([r["nD"] for r in dc], float)
d_gb = np.array([r["mem_gib"] for r in dc], float)
g_nd = np.array([r["nD"] for r in gn], float)
g_gb = np.array([r["mem_gib"] for r in gn], float)
slope = np.polyfit(g_nd, g_gb, 1)[0] * 1000 * 1024          # MiB / 1000 data
j_only = nC * 8 / 2 ** 20 * 1000                            # MiB / 1000 data

fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# the Jacobian alone, as a line through the conventional intercept
b = np.polyfit(g_nd, g_gb, 1)[1]
xs = np.linspace(0, d_nd.max() * 1.05, 50)
ax.plot(xs, b + j_only / 1024 * xs / 1000, "--", color=GN, lw=1.2, alpha=0.55,
        zorder=2)
ax.text(d_nd.max() * 0.99, b + j_only / 1024 * d_nd.max() / 1000 + 0.12,
        "stored Jacobian alone\n$n_D\\times n_C\\times 8$ bytes",
        fontsize=8.2, color=GN, ha="right", va="bottom", alpha=0.85)

ax.plot(g_nd, g_gb, "-s", color=GN, lw=2.0, ms=6, mec="white", mew=1.0,
        zorder=4, label="stored Jacobian (host memory)")
ax.plot(d_nd, d_gb, "-o", color=OURS, lw=2.0, ms=6, mec="white", mew=1.0,
        zorder=4, label="this work (device memory)")

ax.annotate(f"{slope:.0f} MiB per 1000 data",
            xy=(g_nd[-2], g_gb[-2]), xytext=(-10, 26),
            textcoords="offset points", fontsize=9.0, color=GN,
            ha="right", fontweight="bold")
ax.annotate(f"flat: {d_gb[0]:.3f} GiB at every point",
            xy=(d_nd[4], d_gb[4]), xytext=(0, -24),
            textcoords="offset points", fontsize=9.0, color=OURS,
            ha="center", fontweight="bold")

ax.set_xlabel("number of measurements $n_D$   "
              "(mesh and 342 injections held fixed)", fontsize=10, color=INK)
ax.set_ylabel("peak memory (GiB)", fontsize=10, color=INK)
ax.set_xlim(0, d_nd.max() * 1.06)
ax.set_ylim(0, max(g_gb.max(), d_gb.max()) * 1.28)
ax.grid(True, color=RULE, lw=0.6, alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(labelsize=8.5, colors=MUTE, length=3)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(RULE)
ax.legend(loc="upper left", frameon=False, fontsize=9.0)
ax.text(0.0, 1.03, f"{nC:,} active cells · 342 injections · "
        f"$n_D$ from {int(d_nd[0])} to {int(d_nd[-1])}",
        transform=ax.transAxes, fontsize=9, color=MUTE, va="bottom")

fig.tight_layout()
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print(f"saved {OUT}")
print(f"  slope GN {slope:.1f} MiB/1000 | J alone {j_only:.1f} | "
      f"ours {np.polyfit(d_nd, d_gb, 1)[0]*1000*1024:+.2f}")
