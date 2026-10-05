"""
inv_tutorial3d_vanilla.py — el tutorial OFICIAL de SimPEG "3D DC Resistivity
Inversion" (user-tutorials/05-dcr/inv_dcr_3d, dos esferas sinteticas),
VERBATIM menos las celdas de ploteo, con DH parametrizado (el tutorial usa 25).

Receta calcada del notebook: unc = 1e-7 + 10%|d|; TreeMesh dh con
refine_surface [0,4,4] + refine_points [6,6,4]; sigmaMap = InjectActive(1e-8)
* Exp; Simulation3DNodal storeJ=True (solver default = Pardiso);
WeightedLeastSquares ls=100; InexactGaussNewton(maxIter=40, maxIterLS=20,
cg_maxiter=30, cg_rtol=1e-3); directives = UpdateSensitivityWeights(every,
1e-2) + UpdatePreconditioner(every) + BetaEstimate_ByEig(100) +
BetaSchedule(2,2) + TargetMisfit(chifact=1).

Env: DH (def 60). Salida: inv_tutorial3d_vanilla_dh{DH}.npz + JSON en stdout.
Correr (CWD _gpubench, env simpeg311): python -u inv_tutorial3d_vanilla.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import tarfile
import time

import numpy as np
import psutil

from simpeg.electromagnetics.static import resistivity as dc  # noqa: F401
from simpeg.electromagnetics.static.utils.static_utils import (
    apparent_resistivity_from_voltage)
from simpeg.utils.io_utils.io_utils_electromagnetics import read_dcip_xyz
from simpeg.utils import download, model_builder
from simpeg import (maps, data_misfit, regularization, optimization,
                    inverse_problem, inversion, directives)
from discretize import TreeMesh
from discretize.utils import active_from_xyz

HERE = (os.environ.get("DCTORCH_BENCH_OUT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "_gpubench")))
DH = float(os.environ.get("DH", "60"))

# ---------- datos del tutorial (descarga con cache local) ----------
dir_path = os.path.join(HERE, "inv_dcr_3d_files") + os.path.sep
if not os.path.exists(os.path.join(dir_path, "topo_xyz.txt")):
    data_source = ("https://github.com/simpeg/user-tutorials/raw/main/assets/"
                   "05-dcr/inv_dcr_3d_files.tar.gz")
    downloaded_data = download(data_source, folder=HERE, overwrite=True)
    os.makedirs(dir_path, exist_ok=True)
    with tarfile.open(downloaded_data, "r") as tar:
        tar.extractall(dir_path)     # el tar trae los archivos en la raiz
topo_xyz = np.loadtxt(dir_path + "topo_xyz.txt")
dc_data, out_dict = read_dcip_xyz(
    dir_path + "dc_data.xyz", "volt", data_header="V/A",
    uncertainties_header="UNCERT", is_surface_data=False,
    dict_headers=["LINEID"])

# (celda 23) incertidumbres del tutorial
dc_data.standard_deviation = 1e-7 + 0.1 * np.abs(dc_data.dobs)

# ---------- malla (celda 25, dh parametrizado) ----------
dom_width_x = dom_width_y = 8000.0
dom_width_z = 4000.0
nbcx = 2 ** int(np.round(np.log(dom_width_x / DH) / np.log(2.0)))
nbcy = 2 ** int(np.round(np.log(dom_width_y / DH) / np.log(2.0)))
nbcz = 2 ** int(np.round(np.log(dom_width_z / DH) / np.log(2.0)))
mesh = TreeMesh([[(DH, nbcx)], [(DH, nbcy)], [(DH, nbcz)]], x0="CCN",
                diagonal_balance=True)
mesh.origin = mesh.origin + np.r_[0.0, 0.0, topo_xyz[:, -1].max()]
k = np.sqrt(np.sum(topo_xyz[:, 0:2] ** 2, axis=1)) < 1200
mesh.refine_surface(topo_xyz[k, :], padding_cells_by_level=[0, 4, 4],
                    finalize=False)
unique_locations = dc_data.survey.unique_electrode_locations
mesh.refine_points(unique_locations, padding_cells_by_level=[6, 6, 4],
                   finalize=False)
mesh.finalize()
active_cells = active_from_xyz(mesh, topo_xyz)
n_active = int(np.sum(active_cells))
dc_data.survey.drape_electrodes_on_topography(mesh, active_cells,
                                              topo_cell_cutoff="top")
print(f"[mesh] dh={DH:g} | cells {mesh.n_cells:,} | active {n_active:,} | "
      f"nodes {mesh.n_nodes:,} | nD {dc_data.survey.nD}", flush=True)

# ---------- mapas y modelos (celdas 31-33) ----------
log_conductivity_map = maps.InjectActiveCells(
    mesh, active_cells, 1e-8) * maps.ExpMap(nP=n_active)
apparent_conductivities = 1 / apparent_resistivity_from_voltage(
    dc_data.survey, dc_data.dobs)
median_conductivity = np.median(apparent_conductivities)
starting_conductivity_model = np.log(median_conductivity) * np.ones(n_active)
reference_conductivity_model = starting_conductivity_model.copy()

# ---------- simulacion + problema inverso (celdas 35-45, verbatim) ----------
dc_simulation = dc.simulation.Simulation3DNodal(
    mesh, survey=dc_data.survey, sigmaMap=log_conductivity_map, storeJ=True)
dmis_L2 = data_misfit.L2DataMisfit(simulation=dc_simulation, data=dc_data)
reg_L2 = regularization.WeightedLeastSquares(
    mesh, active_cells=active_cells, length_scale_x=100.0,
    length_scale_y=100.0, length_scale_z=100.0,
    reference_model=reference_conductivity_model)
opt_L2 = optimization.InexactGaussNewton(maxIter=40, maxIterLS=20,
                                         cg_maxiter=30, cg_rtol=1e-3)
inv_prob_L2 = inverse_problem.BaseInvProblem(dmis_L2, reg_L2, opt_L2)
sensitivity_weights = directives.UpdateSensitivityWeights(
    every_iteration=True, threshold_value=1e-2)
update_jacobi = directives.UpdatePreconditioner(update_every_iteration=True)
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=100)
beta_schedule = directives.BetaSchedule(coolingFactor=2.0, coolingRate=2)
target_misfit = directives.TargetMisfit(chifact=1.0)


class IterTimer(directives.InversionDirective):
    """Cronometro por iteracion: (iter, t_acum, phi_d, phi_m, beta)."""

    def initialize(self):
        self.rows = []
        self.t0 = time.time()

    def endIter(self):
        self.rows.append((self.opt.iter, time.time() - self.t0,
                          float(self.invProb.phi_d),
                          float(self.invProb.phi_m),
                          float(self.invProb.beta)))


iter_timer = IterTimer()
directives_list_L2 = [sensitivity_weights, update_jacobi, starting_beta,
                      beta_schedule, target_misfit, iter_timer]
inv_L2 = inversion.BaseInversion(inv_prob_L2, directives_list_L2)

t0 = time.time()
recovered = inv_L2.run(starting_conductivity_model)
t_inv = time.time() - t0
phid = float(dmis_L2(recovered))
nD = int(dc_data.survey.nD)
chi = phid / nD
rss = psutil.Process().memory_info().peak_wset / 2**30
n_it = int(opt_L2.iter)

# modelo verdadero (celda 54) para score de recuperacion
true_model = 1e-2 * np.ones(n_active)
cc = mesh.cell_centers[active_cells, :]
true_model[model_builder.get_indices_sphere(
    np.r_[-300.0, 0.0, 100.0], 165.0, cc)] = 1e-1
true_model[model_builder.get_indices_sphere(
    np.r_[300.0, 0.0, 100.0], 165.0, cc)] = 1e-3
corr_true = float(np.corrcoef(np.log(true_model), recovered)[0, 1])

print(f"\n[tutorial-vanilla dh={DH:g}] {t_inv:.1f}s | {n_it} iters | "
      f"phi_d {phid:.1f} (target {nD}) | chi {chi:.3f} | "
      f"corr(log) vs verdad {corr_true:.4f} | RAM {rss:.2f} GiB", flush=True)
out = os.path.join(HERE, f"inv_tutorial3d_vanilla_dh{DH:g}.npz")
np.savez_compressed(out, m_log_sigma=recovered, t=t_inv, n_iters=n_it,
                    phid=phid, chi=chi, corr_true=corr_true, nC=n_active,
                    true_log_sigma=np.log(true_model), dh=DH,
                    iter_log=np.array(iter_timer.rows))
print("JSON: " + json.dumps(dict(dh=DH, engine="tutorial-GN-pardiso",
                                 t=round(t_inv, 1), iters=n_it,
                                 chi=round(chi, 3),
                                 corr_true=round(corr_true, 4),
                                 nC=n_active, ram_gib=round(rss, 2))))
print(f"saved {out}", flush=True)
