"""
plot_tutorial_slices_dh25.py — panel de modelos del benchmark dos-esferas
dh=25 (verdad / tutorial-GN / This engine, F64 / F32), cortes y=0 como la celda 57
del tutorial, en DOS variantes: con sensitivity weights (tag "") y sin
(_nosw). Titulos con padding y filas separadas (hspace) para que no se peguen
a los paneles. Lee npz de paper/benchmarks/results, escribe en
paper/benchmarks/figures. Metrica por esfera (contraste medio) a stdout.
Correr (CWD _gpubench, env simpeg311): python plot_tutorial_slices_dh25.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

from simpeg.utils import model_builder
from simpeg import maps
from simpeg.utils.io_utils.io_utils_electromagnetics import read_dcip_xyz
from discretize import TreeMesh
from discretize.utils import active_from_xyz

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), r"paper\benchmarks"))
DATA = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "figures")
DH = 25.0

dir_path = os.path.join(REPO, "..", "..", "examples", "inv_dcr_3d_files") + os.path.sep
topo_xyz = np.loadtxt(dir_path + "topo_xyz.txt")
dc_data, _ = read_dcip_xyz(dir_path + "dc_data.xyz", "volt",
                           data_header="V/A", uncertainties_header="UNCERT",
                           is_surface_data=False, dict_headers=["LINEID"])

nbc = lambda w: 2 ** int(np.round(np.log(w / DH) / np.log(2.0)))  # noqa: E731
mesh = TreeMesh([[(DH, nbc(8000.))], [(DH, nbc(8000.))], [(DH, nbc(4000.))]],
                x0="CCN", diagonal_balance=True)
mesh.origin = mesh.origin + np.r_[0.0, 0.0, topo_xyz[:, -1].max()]
k = np.sqrt(np.sum(topo_xyz[:, 0:2] ** 2, axis=1)) < 1200
mesh.refine_surface(topo_xyz[k, :], padding_cells_by_level=[0, 4, 4],
                    finalize=False)
mesh.refine_points(dc_data.survey.unique_electrode_locations,
                   padding_cells_by_level=[6, 6, 4], finalize=False)
mesh.finalize()
active = active_from_xyz(mesh, topo_xyz)
nC = int(active.sum())
cc = mesh.cell_centers[active, :]

true_sigma = 1e-2 * np.ones(nC)
iC = model_builder.get_indices_sphere(np.r_[-300., 0., 100.], 165., cc)
iR = model_builder.get_indices_sphere(np.r_[300., 0., 100.], 165., cc)
true_sigma[iC] = 1e-1
true_sigma[iR] = 1e-3
iBG = ~(iC | iR)

plotting_map = maps.InjectActiveCells(mesh, active, np.nan)
norm = LogNorm(vmin=1e-3, vmax=1e-1)
zmax = topo_xyz[:, -1].max()

van = np.load(os.path.join(DATA, "inv_tutorial3d_vanilla_dh25.npz"))

for tag in ("", "_nosw"):
    f64 = np.load(os.path.join(DATA,
                               f"inv_tutorial3d_dctorch_f64_dh25{tag}.npz"))
    f32 = np.load(os.path.join(DATA,
                               f"inv_tutorial3d_dctorch_f32_dh25{tag}.npz"))
    sw = "with sensitivity weights" if tag == "" else "no sensitivity weights"
    models = {
        "True model": np.log(true_sigma),
        f"SimPEG GN ({float(van['t']):.0f} s)": van["m_log_sigma"],
        f"This engine, F64, {sw} ({float(f64['t']):.0f} s)": -f64["m_log_rho"],
        f"This engine, F32, {sw} ({float(f32['t']):.0f} s)": -f32["m_log_rho"],
    }

    print(f"=== variante {tag or 'sensw'} ===")
    print(f"{'modelo':>42} {'contraste C':>11} {'contraste R':>11}")
    for name, m in models.items():
        l10 = m / np.log(10)
        bg = l10[iBG].mean()
        print(f"{name:>42.42} {l10[iC].mean()-bg:11.2f} "
              f"{l10[iR].mean()-bg:11.2f}")

    fig, axs = plt.subplots(2, 2, figsize=(13, 9.2))
    fig.subplots_adjust(hspace=0.42, wspace=0.18, top=0.94)
    for ax, (name, m) in zip(axs.ravel(), models.items()):
        mesh.plot_slice(plotting_map * np.exp(m), ax=ax, normal="Y",
                        ind=int(len(mesh.h[1]) / 2), grid=False,
                        pcolor_opts={"cmap": mpl.cm.RdYlBu_r, "norm": norm})
        for xc, sty in [(-300., "k-"), (300., "k--")]:
            th = np.linspace(0, 2 * np.pi, 100)
            ax.plot(xc + 165 * np.cos(th), 100 + 165 * np.sin(th), sty,
                    lw=1.2)
        ax.set_title(name, fontsize=12, pad=12)
        ax.set_xlim([-1200, 1200])
        ax.set_ylim([zmax - 1200, zmax])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.RdYlBu_r)
    fig.colorbar(sm, ax=axs, shrink=0.7, pad=0.02,
                 label=r"$\sigma$ (S/m)")
    out = os.path.join(OUT, f"fig_spheres_models{tag}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}\n")
