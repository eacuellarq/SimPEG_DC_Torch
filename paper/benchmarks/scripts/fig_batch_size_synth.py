"""fig_batch_size_synth.py — what a batch size costs, on the public synthetic survey.

The engine forms the fields for several current injections in one solve, and how
many go into that solve is a free parameter. (a) Time to form every field once;
(b) peak device memory; both against the batch size. The circle marks the
fastest batch; the shaded band is the range explored in earlier work.

Same measurements as fig_batch_size.py (bench_rhs_width.py, synthetic survey with
2470 injections and 91,592 nodes, single precision, solve only, one cold process
per point), drawn without in-panel text.

Outputs: ../figures/fig_batch_size_synth.{png,pdf}
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
INK, MUTE, RULE, SHADE = "#1a1f28", "#6b7280", "#c8ccd2", "#8a8f96"
ENG = "#2a5ba6"

# batch size, batches, total solve time (s), ms per injection, NVML peak (GiB)
SYN = np.array([
    [1, 2470, 4.142, 1.677, 0.172], [2, 1235, 2.611, 1.057, 0.014],
    [4, 618, 1.447, 0.586, 0.012], [8, 309, 1.019, 0.413, 0.014],
    [16, 155, 0.699, 0.283, 0.014], [32, 78, 0.645, 0.261, 0.049],
    [64, 39, 0.650, 0.263, 0.131], [128, 20, 0.767, 0.310, 0.283],
    [256, 10, 0.880, 0.356, 0.541], [512, 5, 1.045, 0.423, 1.068],
    [1024, 3, 1.416, 0.573, 2.143], [2470, 1, 1.347, 0.546, 4.264]])
x, t, g = SYN[:, 0], SYN[:, 2], SYN[:, 4]
best = int(np.argmin(t))

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.6), dpi=300, sharex=True,
                             gridspec_kw=dict(hspace=0.28))
fig.patch.set_facecolor("white")

for ax, y, ylab, title in ((a1, t, "time to form every field once (s)", "(a) Time"),
                           (a2, g, "peak device memory (GiB)", "(b) Memory")):
    ax.axvspan(1, 200, color=SHADE, alpha=0.10, lw=0, zorder=0)
    ax.plot(x, y, "-", color=ENG, lw=1.5, marker="o", ms=4.5, mec="white", mew=0.8, zorder=3)
    ax.plot([x[best]], [y[best]], "o", ms=10, mfc="none", mec=ENG, mew=1.3, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 3400)
    ax.set_ylabel(ylab, color=INK)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    ax.grid(True, which="major", color=RULE, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTE, length=2.5, width=0.6, which="both")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
        ax.spines[sp].set_linewidth(0.6)

a1.set_ylim(0.4, 6)
a1.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5]))
a1.yaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "5"]))
a1.yaxis.set_minor_locator(NullLocator())
a2.set_ylim(0.008, 8)
a2.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1]))
a2.yaxis.set_major_formatter(FixedFormatter(["0.01", "0.1", "1"]))
a2.yaxis.set_minor_locator(NullLocator())
a2.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 256, 1024]))
a2.xaxis.set_major_formatter(FixedFormatter(["1", "4", "16", "64", "256", "1024"]))
a2.xaxis.set_minor_locator(NullLocator())
a2.set_xlabel("batch size (current injections solved together)", color=INK)

os.makedirs(FIG, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"fig_batch_size_synth.{ext}"), facecolor="white", bbox_inches="tight")
print("saved", os.path.join(FIG, "fig_batch_size_synth.png"))
print(f"best batch {int(x[best])}: one at a time {t[0]/t[best]:.2f}x, all at once {t[-1]/t[best]:.2f}x")
