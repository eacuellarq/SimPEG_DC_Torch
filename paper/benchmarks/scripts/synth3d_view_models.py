"""synth3d_view_models.py — sections and a depth slice of every finished benchmark
model next to the truth, with the true bodies outlined."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch  # noqa: F401  (before simpeg/pydiso: MKL DLL clash)

import synth3d_common as S
import synth3d_bench as Bn

ONLY = os.environ.get("VIEW_ONLY", "")          # substring filter on file names
FILES = [x for x in os.environ.get("VIEW_FILES", "").split(",") if x]   # exact file names
runs = []
for f in sorted(glob.glob(os.path.join(Bn.OUT, "bench_*_dh*.npz"))):
    if ONLY and ONLY not in os.path.basename(f):
        continue
    if FILES and os.path.basename(f) not in FILES:
        continue
    z = np.load(f)
    if z["m_log_rho"].size:
        runs.append((json.loads(str(z["row"])), z["m_log_rho"]))
runs.sort(key=lambda t: (-t[0]["dh"], t[0]["engine"], t[0].get("tag", "")))

UNITS = ("cover", "cap", "vein", "stock", "deep")
kw = dict(cmap="Spectral", vmin=np.log10(5), vmax=np.log10(5000), shading="auto")


def views():
    p_deep, p_stock = np.array(S.DEEP["c"]), np.array(S.STOCK["c"])
    u = (p_stock - p_deep) / np.linalg.norm(p_stock - p_deep)
    half = 0.5 * np.linalg.norm(p_stock - p_deep) + 400.0
    _, dd, _ = S._vein_frame()
    dd = dd[:2] / np.linalg.norm(dd[:2])
    out = []
    for name, c0, d, span in (("deep conductor -> vein -> stock", 0.5 * (p_deep + p_stock), u, half),
                              ("across the vein", np.array(S.VEIN["p0"]), dd, 1300.0)):
        s = np.arange(-span, span + 1, 10.0)
        xs, ys = c0[0] + s * d[0], c0[1] + s * d[1]
        zt = S.topo_z(xs, ys)
        zz = np.arange(zt.min() - 650.0, zt.max() + 5, 8.0)
        SG, ZG = np.meshgrid(s, zz)
        P = np.c_[np.broadcast_to(xs, SG.shape).ravel(), np.broadcast_to(ys, SG.shape).ravel(), ZG.ravel()]
        out.append(("section", name, SG, ZG, P, ZG > zt[None, :]))
    gx, gy = np.arange(-1500.0, 1501, 10.0), np.arange(-1100.0, 1101, 10.0)
    X, Y = np.meshgrid(gx, gy)
    P = np.c_[X.ravel(), Y.ravel(), (S.topo_z(X, Y) - 120.0).ravel()]
    out.append(("plan", "120 m below the surface", X, Y, P, np.zeros(X.shape, bool)))
    return out


V = views()
fig, axs = plt.subplots(len(runs) + 1, len(V), figsize=(7.2 * len(V), 2.9 * (len(runs) + 1)),
                        squeeze=False, gridspec_kw=dict(width_ratios=[1.35, 1.2, 1.0]))
unit_grids = []
for j, (kind, name, A1, A2, P, air) in enumerate(V):
    t = S.log_rho(P).reshape(A1.shape) / np.log(10)
    t[air] = np.nan
    axs[0, j].pcolormesh(A1, A2, t, **kw)
    m = S.unit_masks(P)
    ug = np.zeros(len(P))
    for k, un in enumerate(UNITS, start=1):
        ug[m[un] & (ug == 0) if un != "cover" else m[un]] = k if un != "cover" else 0
    unit_grids.append(np.where(air.ravel(), np.nan, ug).reshape(A1.shape))
    axs[0, j].set_title(f"truth | {name}", fontsize=9)

mesh_cache = {}
for i, (row, m_lr) in enumerate(runs, start=1):
    dh = row["dh"]
    if dh not in mesh_cache:
        sv = S.build_survey()
        mesh_cache[dh] = S.build_mesh(dh, sv, S.topo_points())
    mesh, active = mesh_cache[dh]
    full = np.full(mesh.n_cells, np.nan)
    full[active] = m_lr / np.log(10)
    who = ("this engine" if row["engine"] == "dctorch" else f"SimPEG {row.get('mode')}")
    who += f" [target chi2 {row['target_chi2']:g}]"
    for j, (kind, name, A1, A2, P, air) in enumerate(V):
        img = full[mesh.point2index(P)].reshape(A1.shape)
        img[air] = np.nan
        axs[i, j].pcolormesh(A1, A2, img, **kw)
        axs[i, j].set_title(f"{who} dh={dh:g} | chi2 {row['chi2_final']} (truth {row['target_chi2']}) | "
                            f"{row['t_total_after_mesh_s']/60:.1f} min | corr {row.get('corr_log10_core')}",
                            fontsize=8)
for i in range(len(runs) + 1):
    for j, (kind, name, A1, A2, P, air) in enumerate(V):
        ax = axs[i, j]
        ax.contour(A1, A2, unit_grids[j], levels=np.arange(0.5, len(UNITS) + 1), colors="k",
                   linewidths=0.6)
        ax.set_aspect("equal")
        if kind == "plan":
            ax.set(xlabel="Easting (m)", ylabel="Northing (m)")
        else:
            ax.set(xlabel="distance (m)", ylabel="elevation (m)")
sm = plt.cm.ScalarMappable(cmap="Spectral", norm=plt.Normalize(np.log10(5), np.log10(5000)))
fig.colorbar(sm, ax=axs, shrink=0.5, label="log10 resistivity (ohm-m)")
png = os.path.join(Bn.OUT, os.environ.get("VIEW_PNG", "bench_models_view.png"))
fig.savefig(png, dpi=110, bbox_inches="tight")
print(f"[fig] {png} | runs drawn: " + ", ".join(f"{r['engine']} dh={r['dh']:g}" for r, _ in runs))
