"""
plot_dcip_synth.py — la figura del peldano 2: mismo dato sintetico, el flujo
SECUENCIAL (DC + IP linealizado, lo unico que permite SimPEG stock) contra el
CONJUNTO no-lineal (rho, eta) de dctorch, ambos contra la verdad.

  fig_dcip_synth_models.png  — 3x2: verdad / secuencial / conjunto
  fig_dcip_synth_fit.png     — ajuste de los DOS datasets bajo la fisica
                               EXACTA (el diagnostico que delata al
                               secuencial: no existe un par (sigma,eta) que
                               explique los dos a la vez)

Env: TAG (sufijo del npz, def "") | CWD scripts/, env simpeg311.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from dcip_synth_common import CHRG, COND, REPO, RESI

DATA = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "figures")
TAG = os.environ.get("TAG", "")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e8e7e4"
BLUE, ORANGE, GRAY = "#2a78d6", "#e0651a", "#52514e"

d = np.load(os.path.join(DATA, f"inv_dcip_synth_f64{TAG}.npz"))
import discretize
mesh = discretize.TensorMesh([d["hx"], d["hz"]], origin=d["origin"])
xlim = (-10.0, 240.0)
zlim = (-45.0, 3.0)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.linewidth": 0.6,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.frameon": False, "legend.fontsize": 7,
    "savefig.dpi": 400, "savefig.facecolor": "white"})


def outline(ax):
    for b, c in ((COND, "w"), (CHRG, "w"), (RESI, "w")):
        ax.add_patch(mpl.patches.Rectangle(
            (b["x"][0], b["z"][0]), b["x"][1] - b["x"][0],
            b["z"][1] - b["z"][0], fill=False, ec=c, lw=0.8,
            linestyle=(0, (3, 2))))


# --------------------------------------------------------------- 1. modelos
rho_rows = [("Verdad", d["m_rho_true"]),
            ("Secuencial: inversion DC sola", d["m_rho_seq_inst"]),
            ("Conjunto no-lineal (rho, eta)", d["m_rho_joint"])]
eta_rows = [("Verdad", d["eta_true"]),
            ("Secuencial: IP linealizado (Seigel)", d["eta_seq"]),
            ("Conjunto no-lineal (rho, eta)", d["eta_joint"])]
nr = mpl.colors.Normalize(vmin=np.log10(np.exp(d["m_rho_true"])).min(),
                          vmax=np.log10(np.exp(d["m_rho_true"])).max())
ne = mpl.colors.Normalize(vmin=0.0, vmax=float(d["eta_true"].max()))

fig, axs = plt.subplots(3, 2, figsize=(9.2, 7.0))
fig.subplots_adjust(hspace=0.55, wspace=0.22)
for j, (rows, norm, cmap, lab, conv) in enumerate((
        (rho_rows, nr, mpl.cm.viridis, r"$\log_{10}\rho$ ($\Omega$m)",
         lambda v: np.log10(np.exp(v))),
        (eta_rows, ne, mpl.cm.plasma, r"$\eta$ (V/V)", lambda v: v))):
    for i, (name, v) in enumerate(rows):
        ax = axs[i, j]
        ax.set_rasterization_zorder(1)
        mesh.plot_image(conv(v), ax=ax,
                        pcolor_opts={"cmap": cmap, "norm": norm})
        outline(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)
        ax.set_title(name, fontsize=8, pad=4)
        ax.set_ylabel("z (m)" if j == 0 else "")
        ax.set_xlabel("x (m)" if i == 2 else "")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axs[:, j], shrink=0.7, pad=0.02, label=lab)
out = os.path.join(OUT, f"fig_dcip_synth_models{TAG}.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# ------------------------------------------------- 2. ajuste de los datasets
fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.9))
r_seq_dc = (d["pred_seq_dc"] - d["dobs_dc"]) / d["std_dc"]
r_jt_dc = (d["pred_joint_dc"] - d["dobs_dc"]) / d["std_dc"]
r_seq_ip = (d["pred_seq_ip"] - d["dobs_ip"]) / d["std_ip"]
r_jt_ip = (d["pred_joint_ip"] - d["dobs_ip"]) / d["std_ip"]

ax = axs[0]
CHI_S = d["chi_seq_used"] if "chi_seq_used" in d else d["chi_seq_b"]
ax.plot(d["dobs_ip"], d["pred_seq_ip"], "o", ms=3, mfc="none", mew=0.8,
        mec=ORANGE, label=f"secuencial ($\\chi^2$ {float(CHI_S[1]):.1f})")
ax.plot(d["dobs_ip"], d["pred_joint_ip"], "o", ms=3, mfc="none", mew=0.8,
        mec=BLUE, label=f"conjunto ($\\chi^2$ {float(d['chi_joint'][1]):.2f})")
lo, hi = float(d["dobs_ip"].min()), float(d["dobs_ip"].max())
ax.plot([lo, hi], [lo, hi], "-", color=INK2, lw=0.7)
ax.set_xlabel(r"$\eta_a$ observada (V/V)")
ax.set_ylabel(r"$\eta_a$ predicha (V/V)")
ax.set_title("dato IP bajo la fisica EXACTA", fontsize=8)
ax.legend(loc="upper left")

ax = axs[1]
for r, c, n in ((r_seq_ip, ORANGE, "secuencial"), (r_jt_ip, BLUE, "conjunto")):
    ax.hist(r, bins=24, histtype="step", color=c, lw=1.1, label=n)
ax.axvline(0, color=INK2, lw=0.6, dashes=(2, 2))
ax.set_xlabel(r"residuo normalizado IP $(d_{pred}-d_{obs})/\sigma$")
ax.set_ylabel("cuentas")
ax.set_title("residuos IP", fontsize=8)
ax.legend()

ax = axs[2]
for r, c, n in ((r_seq_dc, ORANGE, "secuencial"), (r_jt_dc, BLUE, "conjunto")):
    ax.hist(r, bins=24, histtype="step", color=c, lw=1.1, label=n)
ax.axvline(0, color=INK2, lw=0.6, dashes=(2, 2))
ax.set_xlabel(r"residuo normalizado DC $(d_{pred}-d_{obs})/\sigma$")
ax.set_title("residuos DC", fontsize=8)
ax.legend()
for ax in axs:
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.6)
out = os.path.join(OUT, f"fig_dcip_synth_fit{TAG}.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")
print(f"chi2 IP  secuencial {float(CHI_S[1]):.3f} | conjunto "
      f"{float(d['chi_joint'][1]):.3f}")
print(f"chi2 DC  secuencial {float(CHI_S[0]):.3f} | conjunto "
      f"{float(d['chi_joint'][0]):.3f}")
