"""
plot_century_dcip.py — Century 46800E, dato publico real: los DOS campos que
salen de UNA inversion conjunta no-lineal (rho, eta) comparados con el
peldano 1 (eta linealizada sobre el fondo DC ya invertido), con la seccion
geologica de Leapfrog superpuesta.

    fig_century_dcip.png

Env: TAG (sufijo del npz, def "") | CWD scripts/, env simpeg311.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from century_dcip_common import IP_SCALE, REPO, build
from century_ip_common import DATA as DATA_IP

DATA = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "figures")
TAG = os.environ.get("TAG", "")
INK2 = "#52514e"


def load_geology():
    with open(os.path.join(os.path.dirname(DATA_IP),
                           "geologic_section.csv")) as fid:
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


survey, _, _, _, _, mesh, dx = build()
geo = load_geology()
j = np.load(os.path.join(DATA, f"inv_century_dcip_f64{TAG}.npz"))
lin = np.load(os.path.join(DATA, "inv_century_ip_dctorch_f64.npz"))
eta_j = j["eta"] * IP_SCALE                        # V/V -> mV/V
eta_l = lin["eta"]                                 # ya en mV/V
corr = float(np.corrcoef(eta_j, eta_l)[0, 1])
print(f"corr(eta conjunta, eta linealizada) = {corr:.4f} | max "
      f"{eta_j.max():.1f} vs {eta_l.max():.1f} mV/V")

xe = survey.unique_electrode_locations[:, 0]
ne = mpl.colors.Normalize(0.0, float(np.percentile(np.r_[eta_j, eta_l], 99.5)))
nr = mpl.colors.Normalize(*np.percentile(np.log10(np.exp(j["m_log_rho"])),
                                         [1, 99]))
panels = [
    (r"Conjunto no-lineal: $\log_{10}\rho$  "
     f"($\\chi^2_{{DC}}$ {float(j['chi_dc']):.2f}, "
     f"{float(j['t']):.0f} s, {int(j['hist'].shape[0])} iters)",
     np.log10(np.exp(j["m_log_rho"])), nr, mpl.cm.viridis,
     r"$\log_{10}\rho$ ($\Omega$m)"),
    (r"Conjunto no-lineal: $\eta$  "
     f"($\\chi^2_{{IP}}$ {float(j['chi_ip']):.2f}, fondo LIBRE)",
     eta_j, ne, mpl.cm.plasma, r"$\eta$ (mV/V)"),
    (f"Peldano 1: $\\eta$ linealizada (fondo DC fijo)  —  corr {corr:.3f}",
     eta_l, ne, mpl.cm.plasma, r"$\eta$ (mV/V)"),
]

fig, axs = plt.subplots(3, 1, figsize=(9.5, 10.2))
fig.subplots_adjust(hspace=0.5)
for ax, (name, v, norm, cmap, lab) in zip(axs, panels):
    ax.set_rasterization_zorder(1)
    mesh.plot_image(v, ax=ax, pcolor_opts={"cmap": cmap, "norm": norm})
    for g in geo:
        ax.plot(g[:, 0], g[:, 1], "w-", lw=1.0, alpha=0.8)
    ax.plot(xe, np.zeros_like(xe), "kv", ms=3)
    ax.set_xlim([xe.min() - 100, xe.max() + 100])
    ax.set_ylim([-500, 30])
    ax.set_title(name, fontsize=11, pad=10)
    ax.set_xlabel("Northing (m)")
    ax.set_ylabel("z (m)")
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 shrink=0.9, pad=0.02, label=lab)
out = os.path.join(OUT, f"fig_century_dcip{TAG}.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")
