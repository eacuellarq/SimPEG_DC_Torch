"""
century_common.py — carga y setup COMPARTIDO del benchmark Century 2D
(linea 46800E del yacimiento Century Zinc, Queensland, Australia; datos
publicos del tutorial Transform-2020 de SimPEG, repo simpeg/transform-2020-
simpeg, licencia MIT). Ambos motores importan de aqui para garantizar mismo
survey/malla/incertidumbres/arranque.

Receta CALCADA del notebook 1-century-dcip-inversion.ipynb:
  - parser del formato UBC-GIF viejo (.OBS con std por dato)
  - TensorMesh: dx=dz=min_spacing/4=25 m, core [xmin,xmax]x[-max_sep/3,0],
    padding factor 1.3, n_pad para llegar a >=1200 m (3x max separacion)
  - rho0 = mediana de resistividad aparente; modelo = log-rho en TODA la
    malla (sin topo: superficie plana z=0)
  - reg alpha_s=1/dx^2, alpha_x=alpha_y=1 | beta0_ratio=1 | cooling 4 cada 2
  - target chifact=1 (usa las std del archivo)
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OBS = os.path.join(HERE, "..", "data_century", "46800E", "46800POT.OBS")
REPO = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), r"paper\benchmarks"))

PAD_FACTOR = 1.3
ALPHA_X = 1.0
BETA0_RATIO = 1.0
COOL, COOLING_RATE = 4.0, 2


def read_century_obs(filename=OBS):
    """Parser del UBC-GIF viejo (del notebook, np.str -> str)."""
    contents = np.genfromtxt(filename, delimiter=" \n", dtype=str)
    n_sources = int(contents[1].split()[0])
    a, b = np.zeros(n_sources), np.zeros(n_sources)
    m_loc, n_loc, dobs, std = [], [], [], []
    row = 2
    for i in range(n_sources):
        src = contents[row].split()
        a[i], b[i], n_rx = float(src[0]), float(src[1]), int(src[2])
        row += 1
        rx = np.array([contents[row + j].split() for j in range(n_rx)],
                      dtype=float)
        row += n_rx
        m_loc.append(rx[:, 0])
        n_loc.append(rx[:, 1])
        dobs.append(rx[:, 2])
        std.append(rx[:, 3])
    return a, b, m_loc, n_loc, dobs, std


def build_survey():
    from simpeg.electromagnetics.static import resistivity as dc
    a, b, m_loc, n_loc, dobs, std = read_century_obs()
    src_list = []
    for i in range(len(a)):
        rx = dc.receivers.Dipole(
            locations_m=np.c_[m_loc[i], np.zeros_like(m_loc[i])],
            locations_n=np.c_[n_loc[i], np.zeros_like(n_loc[i])])
        src_list.append(dc.sources.Dipole(
            [rx], location_a=np.r_[a[i], 0.0], location_b=np.r_[b[i], 0.0]))
    survey = dc.Survey(src_list)
    return survey, np.hstack(dobs), np.hstack(std)


def build_mesh(survey):
    import discretize
    locs = survey.unique_electrode_locations
    spac = np.abs(np.diff(np.unique(locs[:, 0])))
    dx = dz = float(spac.min()) / 4.0
    x0, x1 = locs[:, 0].min(), locs[:, 0].max()
    mid_ab = (survey.locations_a + survey.locations_b) / 2
    mid_mn = (survey.locations_m + survey.locations_n) / 2
    max_sep = float(np.abs(mid_ab - mid_mn)[:, 0].max())
    n_core_x = int(np.ceil((x1 - x0) / dx)) + 8
    n_core_z = int(np.ceil(max_sep / 3.0 / dz))
    # padding geometrico hasta cubrir >= 3*max_sep
    n_pad = 1
    while dx * (PAD_FACTOR ** np.arange(1, n_pad + 1)).sum() < 3 * max_sep:
        n_pad += 1
    hx = [(dx, n_pad, -PAD_FACTOR), (dx, n_core_x), (dx, n_pad, PAD_FACTOR)]
    hz = [(dz, n_pad, -PAD_FACTOR), (dz, n_core_z)]
    mesh = discretize.TensorMesh([hx, hz])
    mesh.origin = np.r_[
        x0 - 4 * dx - mesh.h[0][:n_pad].sum(), -mesh.h[1].sum()]
    return mesh, dx


def starting_halfspace(survey, dobs):
    from simpeg.electromagnetics.static.utils.static_utils import (
        apparent_resistivity_from_voltage)
    rho_app = apparent_resistivity_from_voltage(survey, dobs)
    ok = np.isfinite(rho_app) & (rho_app > 0)
    return float(np.median(rho_app[ok]))
