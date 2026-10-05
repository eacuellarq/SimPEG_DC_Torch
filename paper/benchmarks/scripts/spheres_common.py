"""
spheres_common.py — setup COMPARTIDO del problema dos-esferas (tutorial
oficial user-tutorials/05-dcr/inv_dcr_3d) parametrizado por DH, para los
benchmarks reproducibles (escalado por fases, vanilla por evaluacion, GN
RAM). Mismo dato descargado, misma receta de malla que
inv_tutorial3d_{vanilla,dctorch}.py.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), r"paper\benchmarks"))
DIR = os.path.join(HERE, "inv_dcr_3d_files") + os.path.sep


def build_problem(DH):
    """mesh, survey (drapeado), dobs, std, active, nC del tutorial a DH."""
    from simpeg.utils.io_utils.io_utils_electromagnetics import read_dcip_xyz
    from discretize import TreeMesh
    from discretize.utils import active_from_xyz

    assert os.path.exists(os.path.join(DIR, "topo_xyz.txt")), \
        "correr inv_tutorial3d_vanilla.py primero (descarga el tar)"
    topo_xyz = np.loadtxt(DIR + "topo_xyz.txt")
    dc_data, _ = read_dcip_xyz(
        DIR + "dc_data.xyz", "volt", data_header="V/A",
        uncertainties_header="UNCERT", is_surface_data=False,
        dict_headers=["LINEID"])
    std = 1e-7 + 0.1 * np.abs(dc_data.dobs)

    nbc = lambda w: 2 ** int(np.round(np.log(w / DH) / np.log(2.0)))  # noqa
    mesh = TreeMesh([[(DH, nbc(8000.))], [(DH, nbc(8000.))],
                     [(DH, nbc(4000.))]], x0="CCN", diagonal_balance=True)
    mesh.origin = mesh.origin + np.r_[0.0, 0.0, topo_xyz[:, -1].max()]
    k = np.sqrt(np.sum(topo_xyz[:, 0:2] ** 2, axis=1)) < 1200
    mesh.refine_surface(topo_xyz[k, :], padding_cells_by_level=[0, 4, 4],
                        finalize=False)
    mesh.refine_points(dc_data.survey.unique_electrode_locations,
                       padding_cells_by_level=[6, 6, 4], finalize=False)
    mesh.finalize()
    active = active_from_xyz(mesh, topo_xyz)
    dc_data.survey.drape_electrodes_on_topography(mesh, active,
                                                  topo_cell_cutoff="top")
    return mesh, dc_data.survey, dc_data.dobs, std, active, int(active.sum())


def starting_logrho(survey, dobs):
    from simpeg.electromagnetics.static.utils.static_utils import (
        apparent_resistivity_from_voltage)
    app_cond = 1 / apparent_resistivity_from_voltage(survey, dobs)
    return float(-np.log(np.median(app_cond)))
