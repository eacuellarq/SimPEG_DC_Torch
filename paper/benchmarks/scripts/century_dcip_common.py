"""
century_dcip_common.py — setup del CONJUNTO no-lineal sobre Century 46800E:
un solo survey (la geometria del 46800POT.OBS, receivers "volt") con los DOS
datasets alineados dato a dato.

Verificado en los archivos (probe 2026-07-26): POT.OBS e IP.OBS tienen las
MISMAS 27 fuentes (a,b identicos) y los mismos 151 pares M-N por fuente, pero
el IP los escribe con M y N intercambiados. Eso NO afecta a la cargabilidad
aparente (es un cociente: V1 y V0 cambian de signo a la vez), pero si obliga a
emparejar por par NO ORDENADO {m,n} para alinear los dos vectores de datos.

Unidades: POT en volt (tal cual, como el benchmark DC); IP en mV/V, de modo
que el motor conjunto se construye con ip_scale=1000 (eta adimensional en el
grafo -> dato predicho en mV/V).
"""
import os

import numpy as np

import century_common as cc
from century_ip_common import OBS_DC, OBS_IP

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IP_SCALE = 1000.0                       # eta adimensional -> dato en mV/V


def _key(m, n):
    return np.c_[np.minimum(m, n), np.maximum(m, n)]


def build():
    """(survey volt, dobs_dc, std_dc, dobs_ip, std_ip, mesh, dx) con el IP
    REORDENADO a la ordenacion del survey DC."""
    from simpeg.electromagnetics.static import resistivity as dc

    a1, b1, m1, n1, d1, s1 = cc.read_century_obs(OBS_DC)
    a2, b2, m2, n2, d2, s2 = cc.read_century_obs(OBS_IP)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2), \
        "las fuentes de POT e IP no coinciden"

    src_list, ip_o, ip_s = [], [], []
    for i in range(len(a1)):
        rx = dc.receivers.Dipole(
            locations_m=np.c_[m1[i], np.zeros_like(m1[i])],
            locations_n=np.c_[n1[i], np.zeros_like(n1[i])],
            data_type="volt")
        src_list.append(dc.sources.Dipole(
            [rx], location_a=np.r_[a1[i], 0.0], location_b=np.r_[b1[i], 0.0]))
        k1, k2 = _key(m1[i], n1[i]), _key(m2[i], n2[i])
        assert k1.shape == k2.shape, f"fuente {i}: distinto numero de rx"
        order = []
        used = np.zeros(len(k2), dtype=bool)
        for r in k1:
            hit = np.where(~used & np.all(np.isclose(k2, r), axis=1))[0]
            assert hit.size, f"fuente {i}: par {r} del DC no esta en el IP"
            order.append(hit[0])
            used[hit[0]] = True
        order = np.array(order)
        ip_o.append(np.asarray(d2[i])[order])
        ip_s.append(np.asarray(s2[i])[order])

    survey = dc.Survey(src_list)
    mesh, dx = cc.build_mesh(survey)
    return (survey, np.hstack(d1), np.hstack(s1), np.hstack(ip_o),
            np.hstack(ip_s), mesh, dx)
