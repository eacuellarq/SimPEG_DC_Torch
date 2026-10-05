"""
dcip_synth_common.py — setup COMPARTIDO del experimento sintetico del IP
CONJUNTO NO-LINEAL (peldano 2): dipolo-dipolo 2.5-D con un conductor y un
cuerpo CARGABLE DESPLAZADO de eta GRANDE (el caso minero donde la
linealizacion de Seigel falla).

Los modelos verdaderos son geometricos (indicadores de bloque evaluados en
los centros de celda), asi que la MISMA verdad se puede evaluar en dos
mallas: los datos se generan en una malla FINA (dx/2) y se invierten en la
gruesa -> el crimen inverso queda reducido a error de discretizacion, no
eliminado (declarado).

Convencion de los datos (ver dctorch/dcip2d.py):
  d_DC = V1 = F(sigma(1-eta))   (dc_state="total": el voltaje de estado
        estacionario POLARIZADO, que es el que mide un receptor de dominio
        del tiempo)
  d_IP = (V1 - V0)/V1           (cargabilidad aparente de Seigel EXACTA,
        adimensional V/V)
"""
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                     # paper/benchmarks

# ---- geometria del levantamiento -------------------------------------------
N_ELEC, SPACING, X0 = 24, 10.0, 0.0
N_MAX = 8                                        # dipolo-dipolo n = 1..8

# ---- verdad ----------------------------------------------------------------
RHO_BG, ETA_BG = 100.0, 0.01
COND = dict(x=(70.0, 110.0), z=(-24.0, -8.0), rho=10.0)      # conductor
CHRG = dict(x=(130.0, 175.0), z=(-22.0, -6.0), eta=0.40)     # cargable
RESI = dict(x=(185.0, 215.0), z=(-20.0, -6.0), rho=1000.0)   # resistivo

# ---- ruido -----------------------------------------------------------------
DC_REL, IP_REL, IP_FLOOR = 0.02, 0.05, 0.002     # 2% | 5% + 0.002 V/V
SEED = 20260726


def build_survey():
    """Dipolo-dipolo n=1..N_MAX, receivers 'volt' (el conjunto forma el
    cociente el mismo; el linealizado usa su propio survey)."""
    from simpeg.electromagnetics.static import resistivity as dc
    xe = X0 + SPACING * np.arange(N_ELEC)
    srcs = []
    for i in range(N_ELEC - 1):
        lm, ln = [], []
        for n in range(1, N_MAX + 1):
            jm, jn = i + 1 + n, i + 2 + n
            if jn >= N_ELEC:
                break
            lm.append([xe[jm], 0.0])
            ln.append([xe[jn], 0.0])
        if lm:
            srcs.append(dc.sources.Dipole(
                [dc.receivers.Dipole(np.array(lm), np.array(ln),
                                     data_type="volt")],
                np.r_[xe[i], 0.0], np.r_[xe[i + 1], 0.0]))
    return dc.survey.Survey(srcs)


def build_survey_ip():
    """Mismo survey con data_type='apparent_chargeability' (lo que necesita
    TorchIP2D para aplicar el 1/V_dc de stock)."""
    from simpeg.electromagnetics.static import resistivity as dc
    s = build_survey()
    out = []
    for src in s.source_list:
        rx = src.receiver_list[0]
        out.append(dc.sources.Dipole(
            [dc.receivers.Dipole(rx.locations[0], rx.locations[1],
                                 data_type="apparent_chargeability")],
            src.location[0], src.location[1]))
    return dc.survey.Survey(out)


def build_mesh(survey, refine=1):
    """La receta de malla de century_common con dx = SPACING/4/refine."""
    import discretize
    locs = survey.unique_electrode_locations
    dx = dz = SPACING / 4.0 / refine
    x0, x1 = locs[:, 0].min(), locs[:, 0].max()
    mid_ab = (survey.locations_a + survey.locations_b) / 2
    mid_mn = (survey.locations_m + survey.locations_n) / 2
    max_sep = float(np.abs(mid_ab - mid_mn)[:, 0].max())
    n_core_x = int(np.ceil((x1 - x0) / dx)) + 8 * refine
    n_core_z = int(np.ceil(max_sep / 3.0 / dz))
    pf = 1.3
    n_pad = 1
    while dx * (pf ** np.arange(1, n_pad + 1)).sum() < 3 * max_sep:
        n_pad += 1
    hx = [(dx, n_pad, -pf), (dx, n_core_x), (dx, n_pad, pf)]
    hz = [(dz, n_pad, -pf), (dz, n_core_z)]
    mesh = discretize.TensorMesh([hx, hz])
    mesh.origin = np.r_[x0 - 4 * dx * refine - mesh.h[0][:n_pad].sum(),
                        -mesh.h[1].sum()]
    return mesh, dx


def _box(cc, b):
    return ((cc[:, 0] >= b["x"][0]) & (cc[:, 0] <= b["x"][1])
            & (cc[:, 1] >= b["z"][0]) & (cc[:, 1] <= b["z"][1]))


def true_models(mesh):
    """(m_rho verdadero = log rho INSTANTANEA, eta verdadera) en la malla."""
    cc = mesh.cell_centers
    rho = RHO_BG * np.ones(mesh.nC)
    rho[_box(cc, COND)] = COND["rho"]
    rho[_box(cc, RESI)] = RESI["rho"]
    eta = ETA_BG * np.ones(mesh.nC)
    eta[_box(cc, CHRG)] = CHRG["eta"]
    return np.log(rho), eta


def make_data(device="cuda", nky=11, dc_state="total"):
    """Genera (dobs_dc, std_dc, dobs_ip, std_ip) en la malla FINA (refine=2)
    con la fisica no-lineal exacta, y devuelve tambien la malla gruesa de
    inversion y la verdad en ella.

    `dc_state` fija QUE es el dato DC de la verdad sintetica:
      "total"         -> V1 = F(sigma(1-eta)), el voltaje POLARIZADO
      "instantaneous" -> V0 = F(sigma), la convencion que el flujo
                         linealizado asume implicitamente
    El segundo es el CONTROL que aisla el error de linealizacion puro del
    error de convencion del dato (sin el, los dos van mezclados)."""
    from dctorch import TorchDCIP2D
    survey = build_survey()
    mesh_f, dx_f = build_mesh(survey, refine=2)
    mesh_c, dx_c = build_mesh(survey, refine=1)
    mr_f, eta_f = true_models(mesh_f)

    eng = TorchDCIP2D(mesh_f, survey, nky=nky, device=device,
                      dtype=torch.float64, dc_state=dc_state)
    with torch.no_grad():
        d_dc, d_ip = eng.dpred_from_fields(
            torch.tensor(mr_f, device=device),
            torch.tensor(eta_f, device=device))
    d_dc = d_dc.cpu().numpy()
    d_ip = d_ip.cpu().numpy()
    eng.solver.free()

    rng = np.random.default_rng(SEED)
    std_dc = DC_REL * np.abs(d_dc)
    std_ip = IP_REL * np.abs(d_ip) + IP_FLOOR
    obs_dc = d_dc + std_dc * rng.standard_normal(d_dc.size)
    obs_ip = d_ip + std_ip * rng.standard_normal(d_ip.size)
    mr_c, eta_c = true_models(mesh_c)
    return dict(survey=survey, mesh=mesh_c, dx=dx_c, mesh_fine=mesh_f,
                m_rho_true=mr_c, eta_true=eta_c, clean_dc=d_dc, clean_ip=d_ip,
                dobs_dc=obs_dc, std_dc=std_dc, dobs_ip=obs_ip, std_ip=std_ip,
                nD=int(survey.nD), dc_state=dc_state)


def starting_halfspace(survey, dobs):
    from simpeg.electromagnetics.static.utils.static_utils import (
        apparent_resistivity_from_voltage)
    rho_app = apparent_resistivity_from_voltage(survey, dobs)
    ok = np.isfinite(rho_app) & (rho_app > 0)
    return float(np.median(rho_app[ok]))
