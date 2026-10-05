"""
tutorial_true_chi.py — el diagnostico: chi del MODELO VERDADERO (dos esferas)
a la resolucion DH con las incertidumbres del tutorial (1e-7+10%|d|). Si
chi_true >> 1, la malla no puede representar el dato al ruido asumido y toda
inversion que llegue a chi=1 esta OBLIGADA a inventar estructura (overfit del
error de discretizacion). Env: DH.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from simpeg.utils import model_builder
from simpeg.utils.io_utils.io_utils_electromagnetics import read_dcip_xyz
from discretize import TreeMesh
from discretize.utils import active_from_xyz

from dctorch import TorchDC3D

HERE = (os.environ.get("DCTORCH_BENCH_OUT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "_gpubench")))
DH = float(os.environ.get("DH", "60"))
dir_path = os.path.join(HERE, "inv_dcr_3d_files") + os.path.sep
topo_xyz = np.loadtxt(dir_path + "topo_xyz.txt")
dc_data, _ = read_dcip_xyz(dir_path + "dc_data.xyz", "volt",
                           data_header="V/A", uncertainties_header="UNCERT",
                           is_surface_data=False, dict_headers=["LINEID"])
std = 1e-7 + 0.1 * np.abs(dc_data.dobs)

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
dc_data.survey.drape_electrodes_on_topography(mesh, active,
                                              topo_cell_cutoff="top")
nD = int(dc_data.survey.nD)

true_sigma = 1e-2 * np.ones(nC)
cc = mesh.cell_centers[active, :]
true_sigma[model_builder.get_indices_sphere(
    np.r_[-300., 0., 100.], 165., cc)] = 1e-1
true_sigma[model_builder.get_indices_sphere(
    np.r_[300., 0., 100.], 165., cc)] = 1e-3

eng = TorchDC3D(mesh, dc_data.survey, device="cuda", dtype=torch.float64,
                active_cells=active, val_inactive=np.log(1e8))
with torch.no_grad():
    d = eng.dpred(torch.tensor(-np.log(true_sigma),
                               device="cuda")).cpu().numpy()
r = (d - dc_data.dobs) / std
chi = float(np.mean(r ** 2))
print(f"[dh={DH:g}] nD {nD} | chi2 del modelo VERDADERO = {chi:.2f} | "
      f"|r| p50 {np.percentile(np.abs(r), 50):.2f} "
      f"p90 {np.percentile(np.abs(r), 90):.2f} "
      f"p99 {np.percentile(np.abs(r), 99):.2f}")
