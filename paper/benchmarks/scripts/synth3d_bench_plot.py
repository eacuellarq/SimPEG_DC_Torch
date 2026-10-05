"""synth3d_bench_plot.py — the inversion benchmark as figures and a table.

Reads bench_results.jsonl and the per-run npz (trajectories and final models).
Works on partial results: whatever has finished is drawn.

  bench_trajectories.png  chi2 against wall time after the mesh, one panel per dh,
                          with the level the true model reaches on that mesh
  bench_cost.png          time to the stopping point, peak host memory, peak VRAM
  bench_quality.png       log10-resistivity error in the survey volume and the
                          level recovered inside each geological unit
  bench_sections.png      deep conductor -> vein -> stock section: truth and the
                          final model of every run on its own mesh
  bench_summary.md        the numbers
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch  # noqa: F401  (before simpeg/pydiso: MKL DLL clash)

import synth3d_common as S
import synth3d_bench as Bn

OUT = Bn.OUT
rows = [json.loads(l) for l in open(Bn.RESULTS) if l.strip()]


def label(r):
    if r["engine"] == "dctorch":
        return f"this engine (GPU, L-BFGS, {r['prec']})"
    return {"gn_storej": "SimPEG GN, J stored", "gn": "SimPEG GN, J not stored",
            "bfgs": "SimPEG BFGS, J not stored"}[r["mode"]]


def npz_of(r):
    if r["engine"] == "dctorch":
        return os.path.join(OUT, f"bench_dctorch_{r['prec']}_dh{r['dh']:g}{r.get('tag', '')}.npz")
    return os.path.join(OUT, f"bench_simpeg_{r['mode']}_dh{r['dh']:g}{r.get('tag', '')}.npz")


STYLE = {"this engine": ("k", "-", "o"), "SimPEG GN, J stored": ("#d95f02", "-", "s"),
         "SimPEG GN, J not stored": ("#7570b3", "--", "^"), "SimPEG BFGS": ("#1b9e77", ":", "d")}


def style(lab):
    for k, v in STYLE.items():
        if lab.startswith(k):
            return v
    return ("0.5", "-", "x")


dhs = sorted({r["dh"] for r in rows}, reverse=True)

# ---------------- trajectories ----------------
fig, axs = plt.subplots(1, len(dhs), figsize=(6 * len(dhs), 4.6), squeeze=False)
for ax, dh in zip(axs[0], dhs):
    for r in [r for r in rows if r["dh"] == dh]:
        f = npz_of(r)
        if not os.path.exists(f):
            continue
        h = np.load(f)["hist"]
        if h.size == 0:
            continue
        pre = r["t_total_after_mesh_s"] - (r["t_loop_s"] or 0.0)
        c, ls, mk = style(label(r))
        ax.plot((pre + h[:, 1]) / 60, h[:, 2], ls=ls, color=c, marker=mk, ms=3, label=label(r))
        if r["stop"] not in ("target",):
            ax.annotate(r["stop"], ((pre + h[-1, 1]) / 60, h[-1, 2]), fontsize=7, color=c)
    tgt = Bn.chi2_target(dh)
    if tgt:
        ax.axhline(tgt, color="k", lw=0.8, ls=":")
        ax.text(ax.get_xlim()[1] if ax.lines else 1, tgt, " truth on this mesh", fontsize=7,
                va="bottom", ha="right")
    ax.set(xscale="log", yscale="log", xlabel="wall time after the mesh (min)",
           ylabel="chi2 per datum", title=f"dh = {dh:g} m")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_trajectories.png"), dpi=130)

# ---------------- cost ----------------
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
labs = []
for r in rows:
    if label(r) not in labs:
        labs.append(label(r))
width = 0.8 / max(len(labs), 1)
for j, lab in enumerate(labs):
    c = style(lab)[0]
    for k, (key, ax, ylab) in enumerate((("t_total_after_mesh_s", axs[0], "time to stop (min)"),
                                         ("peak_host_gib", axs[1], "peak host memory (GiB)"),
                                         ("peak_vram_gib", axs[2], "peak device memory (GiB)"))):
        xs, ys = [], []
        for i, dh in enumerate(dhs):
            rr = [r for r in rows if r["dh"] == dh and label(r) == lab]
            if rr:
                v = rr[-1][key] / (60 if key == "t_total_after_mesh_s" else 1)
                xs.append(i + (j - (len(labs) - 1) / 2) * width); ys.append(v)
                if key == "t_total_after_mesh_s" and rr[-1]["stop"] != "target":
                    ax.text(xs[-1], v, rr[-1]["stop"], rotation=90, fontsize=6, ha="center", va="bottom")
        ax.bar(xs, ys, width=width, color=c, label=lab if k == 0 else None)
        ax.set(xticks=range(len(dhs)), xticklabels=[f"dh={d:g}" for d in dhs], ylabel=ylab)
axs[0].set_yscale("log")
axs[0].legend(fontsize=7)
ram = rows[0].get("hardware", {}).get("ram_gib")
if ram:
    axs[1].axhline(ram, color="k", ls=":", lw=0.8)
    axs[1].text(-0.4, ram, f" RAM {ram:g} GiB", fontsize=7, va="bottom")
axs[2].axhline(8.0, color="k", ls=":", lw=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_cost.png"), dpi=130)

# ---------------- quality ----------------
units = ["cover", "cap", "vein", "stock", "deep", "host"]
fig, axs = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1.6]))
for j, lab in enumerate(labs):
    c = style(lab)[0]
    xs, ys = [], []
    for i, dh in enumerate(dhs):
        rr = [r for r in rows if r["dh"] == dh and label(r) == lab and "rms_log10_core_vol" in r]
        if rr:
            xs.append(i + (j - (len(labs) - 1) / 2) * width); ys.append(rr[-1]["rms_log10_core_vol"])
    axs[0].bar(xs, ys, width=width, color=c, label=lab)
axs[0].set(xticks=range(len(dhs)), xticklabels=[f"dh={d:g}" for d in dhs],
           ylabel="RMS log10 error, survey volume")
axs[0].legend(fontsize=7)
ax = axs[1]
k = 0
for r in rows:
    if "units_log10" not in r:
        continue
    c, _, mk = style(label(r))
    u = r["units_log10"]
    ax.plot(np.arange(len(units)) + 0.08 * (k % 6 - 2.5),
            [u[n]["recovered"] if n in u else np.nan for n in units], mk, color=c, ms=5,
            label=f"{label(r)} dh={r['dh']:g}")
    k += 1
if rows and "units_log10" in rows[0]:
    ax.plot(range(len(units)), [rows[0]["units_log10"][n]["true"] if n in rows[0]["units_log10"] else np.nan
                                for n in units], "_", color="r", ms=30, mew=2, label="truth")
ax.set(xticks=range(len(units)), xticklabels=units, ylabel="median log10 rho inside the unit")
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_quality.png"), dpi=130)

# ---------------- sections ----------------
p_deep, p_stock = np.array(S.DEEP["c"]), np.array(S.STOCK["c"])
u = (p_stock - p_deep) / np.linalg.norm(p_stock - p_deep)
half = 0.5 * np.linalg.norm(p_stock - p_deep) + 400.0
sgrid = np.arange(-half, half + 1, 10.0)
xs_, ys_ = 0.5 * (p_deep + p_stock)[0] + sgrid * u[0], 0.5 * (p_deep + p_stock)[1] + sgrid * u[1]
zt = S.topo_z(xs_, ys_)
zgrid = np.arange(zt.min() - 700.0, zt.max() + 5, 10.0)
SG, ZG = np.meshgrid(sgrid, zgrid)
XX, YY = np.broadcast_to(xs_, SG.shape), np.broadcast_to(ys_, SG.shape)
air = ZG > zt[None, :]
done = [r for r in rows if os.path.exists(npz_of(r))]
fig, axs = plt.subplots(len(done) + 1, 1, figsize=(10, 2.6 * (len(done) + 1)), squeeze=False)
truth = S.log_rho(np.c_[XX.ravel(), YY.ravel(), ZG.ravel()]).reshape(SG.shape) / np.log(10)
truth[air] = np.nan
kw = dict(cmap="Spectral", vmin=np.log10(5), vmax=np.log10(5000), shading="auto")
axs[0, 0].pcolormesh(SG, ZG, truth, **kw)
axs[0, 0].set_title("truth", fontsize=9)
meshes = {}
for i, r in enumerate(done, start=1):
    dh = r["dh"]
    if dh not in meshes:
        sv = S.build_survey()
        meshes[dh] = S.build_mesh(dh, sv, S.topo_points())
    mesh, active = meshes[dh]
    full = np.full(mesh.n_cells, np.nan)
    full[active] = np.load(npz_of(r))["m_log_rho"] / np.log(10)
    idx = mesh.point2index(np.c_[XX.ravel(), YY.ravel(), ZG.ravel()])
    img = full[idx].reshape(SG.shape)
    img[air] = np.nan
    axs[i, 0].pcolormesh(SG, ZG, img, **kw)
    axs[i, 0].set_title(f"{label(r)} | dh={dh:g} | chi2 {r['chi2_final']} | "
                        f"{r['t_total_after_mesh_s']/60:.1f} min | stop {r['stop']}", fontsize=9)
for ax in axs[:, 0]:
    ax.set_aspect("equal")
    ax.set_ylabel("elev (m)")
axs[-1, 0].set_xlabel("distance along deep conductor -> vein -> stock (m)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "bench_sections.png"), dpi=120)

# ---------------- table ----------------
lines = ["| dh | run | stop | iters | chi2 final / truth | time after mesh (min) | peak host (GiB) | "
         "peak VRAM (GiB) | RMS log10 | corr |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (-r["dh"], label(r))):
    it = r.get("outers", r.get("iters"))
    lines.append(f"| {r['dh']:g} | {label(r)} | {r['stop']} | {it} | {r['chi2_final']} / {r['target_chi2']} | "
                 f"{r['t_total_after_mesh_s']/60:.1f} | {r['peak_host_gib']} | {r['peak_vram_gib']} | "
                 f"{r.get('rms_log10_core_vol')} | {r.get('corr_log10_core')} |")
open(os.path.join(OUT, "bench_summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
