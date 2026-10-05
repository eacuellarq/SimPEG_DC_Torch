"""
plot_century_3way.py — LA figura: Century 46800E (dato publico real, POT+IP)
invertido con las TRES formulaciones que existen, sobre la misma malla, el
mismo dato y el mismo solver:

  1. SECUENCIAL LINEALIZADO — codigo STOCK de SimPEG de punta a punta
     (inv_century2d_vanilla.py: DC con GN+Pardiso; luego
      inv_century_ip_vanilla.py: IP con ProjectedGNCG + storeJ sobre el fondo
      DC CONGELADO). Es exactamente el flujo que permite SimPEG hoy.
  2. CONJUNTA DOS ESTADOS — sigma_eff = sigma(1-eta), Seigel exacto en dominio
     del tiempo, los dos campos a la vez (inv_century_dcip.py).
  3. COMPLEJA — sigma* = sigma_DC(1-i*m), la formulacion de RES2DINV / cR2 /
     CRTomo (inv_century_cr.py).

Terreno comun para la resistividad: log rho_0, la resistividad DC POLARIZADA,
que las tres producen sin ambiguedad (la secuencial la recupera directamente
del POT; la de dos estados via rho_inf/(1-eta); la compleja es su parametro).
Cargabilidad en mV/V en las tres.

    fig_century_3way.png

CWD scripts/, env simpeg311.
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
xe = survey.unique_electrode_locations[:, 0]

seq_dc = np.load(os.path.join(DATA, "inv_century2d_vanilla.npz"))
seq_ip = np.load(os.path.join(DATA, "inv_century_ip_vanilla.npz"))
jnt = np.load(os.path.join(DATA, "inv_century_dcip_f64.npz"))
crx = np.load(os.path.join(DATA, "inv_century_cr_f64.npz"))

# --- resistividad, todas como log10(rho_0) [rho DC polarizada] -------------
lr_seq = np.log10(np.exp(seq_dc["m_log_rho"]))
lr_jnt = np.log10(np.exp(jnt["m_log_rho"])
                  / (1.0 - np.clip(jnt["eta"], 0, 0.95)))
lr_crx = np.log10(np.exp(crx["log_rho_dc"]))
# --- cargabilidad, todas en mV/V ------------------------------------------
ch_seq = seq_ip["eta"]                        # ya en mV/V
ch_jnt = jnt["eta"] * IP_SCALE
ch_crx = crx["m"] * IP_SCALE

t_seq = float(seq_dc["t"]) + float(seq_ip["t"])
rows = [
    ("1. Secuencial linealizado — SimPEG stock (GN+Pardiso, fondo congelado)",
     lr_seq, ch_seq,
     f"$\\chi^2_{{DC}}$ {float(seq_dc['chi']):.2f} · "
     f"$\\chi^2_{{IP}}$ {float(seq_ip['chi']):.2f} · {t_seq:.0f} s"),
    ("2. Conjunta dos estados  $\\sigma_{eff}=\\sigma(1-\\eta)$  (dctorch, "
     "fondo LIBRE)", lr_jnt, ch_jnt,
     f"$\\chi^2_{{DC}}$ {float(jnt['chi_dc']):.2f} · "
     f"$\\chi^2_{{IP}}$ {float(jnt['chi_ip']):.2f} · "
     f"{float(jnt['t']):.0f} s · {jnt['hist'].shape[0]} it"),
    ("3. Compleja  $\\sigma^*=\\sigma_{DC}(1-im)$  (RES2DINV/cR2, en dctorch)",
     lr_crx, ch_crx,
     f"$\\chi^2_{{DC}}$ {float(crx['chi_r']):.2f} · "
     f"$\\chi^2_{{IP}}$ {float(crx['chi_i']):.2f} · "
     f"{float(crx['t']):.0f} s · {crx['hist'].shape[0]} it"),
]

allr = np.r_[lr_seq, lr_jnt, lr_crx]
nr = mpl.colors.Normalize(*np.percentile(allr, [2, 98]))
ne = mpl.colors.Normalize(
    0.0, float(np.percentile(np.r_[ch_seq, ch_jnt, ch_crx], 99.5)))

fig, axs = plt.subplots(3, 2, figsize=(15.0, 10.6))
fig.subplots_adjust(hspace=0.62, wspace=0.16)
for i, (name, lr, ch, stat) in enumerate(rows):
    for j, (v, norm, cmap, lab) in enumerate((
            (lr, nr, mpl.cm.viridis, r"$\log_{10}\rho_0$ ($\Omega$m)"),
            (ch, ne, mpl.cm.plasma, r"cargabilidad (mV/V)"))):
        ax = axs[i, j]
        ax.set_rasterization_zorder(1)
        mesh.plot_image(v, ax=ax, pcolor_opts={"cmap": cmap, "norm": norm})
        for g in geo:
            ax.plot(g[:, 0], g[:, 1], "w-", lw=1.0, alpha=0.85)
        ax.plot(xe, np.zeros_like(xe), "kv", ms=3)
        ax.set_xlim([xe.min() - 100, xe.max() + 100])
        ax.set_ylim([-500, 30])
        ax.set_xlabel("Northing (m)", fontsize=9)
        ax.set_ylabel("z (m)" if j == 0 else "", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     shrink=0.88, pad=0.02, label=lab)
        ttl = ("resistividad" if j == 0 else "cargabilidad")
        ax.set_title(ttl, fontsize=9, color=INK2, pad=4)
    axs[i, 0].text(0.0, 1.28, name, transform=axs[i, 0].transAxes,
                   fontsize=11.5, fontweight="bold", va="bottom")
    axs[i, 1].text(1.0, 1.28, stat, transform=axs[i, 1].transAxes,
                   fontsize=9.5, color=INK2, ha="right", va="bottom")
fig.suptitle("Century 46800E (dato publico) — las tres formulaciones de IP "
             "sobre la misma malla, dato y solver", fontsize=13, y=0.985)
out = os.path.join(OUT, "fig_century_3way.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}")

print("\n--- acuerdo entre formulaciones (correlacion sobre la malla) ---")
print(f"cargabilidad  secuencial vs dos-estados : "
      f"{np.corrcoef(ch_seq, ch_jnt)[0,1]:.4f}")
print(f"cargabilidad  secuencial vs compleja    : "
      f"{np.corrcoef(ch_seq, ch_crx)[0,1]:.4f}")
print(f"cargabilidad  dos-estados vs compleja   : "
      f"{np.corrcoef(ch_jnt, ch_crx)[0,1]:.4f}")
print(f"log rho_0     secuencial vs dos-estados : "
      f"{np.corrcoef(lr_seq, lr_jnt)[0,1]:.4f}")
print(f"log rho_0     secuencial vs compleja    : "
      f"{np.corrcoef(lr_seq, lr_crx)[0,1]:.4f}")
print(f"log rho_0     dos-estados vs compleja   : "
      f"{np.corrcoef(lr_jnt, lr_crx)[0,1]:.4f}")
print(f"\ncargabilidad max (mV/V): seq {ch_seq.max():.1f} | dos-estados "
      f"{ch_jnt.max():.1f} | compleja {ch_crx.max():.1f} "
      f"(min compleja {ch_crx.min():.1f})")
