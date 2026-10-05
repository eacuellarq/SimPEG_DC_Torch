"""
century_ip_common.py — setup COMPARTIDO de la extension IP sobre Century 2D
(linea 46800E; 46800IP.OBS = cargabilidad aparente en mV/V, formato UBC-GIF
viejo, mismo parser que el DC). Ambos motores (stock y dctorch) importan de
aqui: mismo survey IP, misma malla (la del DC, celda a celda), mismo fondo
sigma = exp(-m_dc) con m_dc = modelo DC recuperado (results/
inv_century2d_dctorch_f64.npz).

Convencion (extraida del tutorial Transform-2020 1-century-dcip-inversion):
  - receivers con data_type="apparent_chargeability" (la sim IP escala por
    1/V_dc); dobs se usa EN mV/V tal cual -> eta recuperada queda en mV/V
    (el forward es lineal, la unidad la fija el dato — practica UBC)
  - std del archivo OBS sin modificar
  - eta0 = 1e-3, cotas [0, inf), cooling 2 cada 1 (difiere del DC: 4 cada 2)
"""
import os

import numpy as np

import century_common as cc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data_century", "46800E")
OBS_DC = os.path.join(DATA, "46800POT.OBS")
OBS_IP = os.path.join(DATA, "46800IP.OBS")
BG_NPZ = os.path.join(os.path.dirname(HERE), "results",
                      "inv_century2d_dctorch_f64.npz")

COOL_IP, COOLING_RATE_IP = 2.0, 1
ETA0 = 1e-3


def _build_survey(obs_path, data_type):
    """build_survey de century_common con ruta explicita (el default de
    read_century_obs quedo ligado a la ruta vieja scripts/century/) y
    data_type parametrizado."""
    from simpeg.electromagnetics.static import resistivity as dc

    a, b, m_loc, n_loc, dobs, std = cc.read_century_obs(obs_path)
    src_list = []
    for i in range(len(a)):
        rx = dc.receivers.Dipole(
            locations_m=np.c_[m_loc[i], np.zeros_like(m_loc[i])],
            locations_n=np.c_[n_loc[i], np.zeros_like(n_loc[i])],
            data_type=data_type)
        src_list.append(dc.sources.Dipole(
            [rx], location_a=np.r_[a[i], 0.0], location_b=np.r_[b[i], 0.0]))
    survey = dc.Survey(src_list)
    return survey, np.hstack(dobs), np.hstack(std)


def build_dc_survey_and_mesh():
    """El survey DC (geometria del POT.OBS) y LA malla del benchmark DC —
    la misma celda a celda donde vive m_dc."""
    survey, _, _ = _build_survey(OBS_DC, "volt")
    mesh, dx = cc.build_mesh(survey)
    return survey, mesh, dx


def build_ip_survey():
    """Survey IP del 46800IP.OBS con data_type='apparent_chargeability'."""
    survey, dobs, std = _build_survey(OBS_IP, "apparent_chargeability")
    return survey, dobs, std


def load_background(mesh):
    """m_dc = log(rho) recuperado del benchmark DC dctorch F64 (chi 0.965);
    valida que la malla sea la misma."""
    d = np.load(BG_NPZ)
    m_dc = d["m_log_rho"]
    assert m_dc.size == mesh.nC, \
        f"malla no coincide con el fondo: {m_dc.size} vs {mesh.nC}"
    assert float(d["dx"]) == float(mesh.h[0].min())
    return m_dc, float(d["chi"])
