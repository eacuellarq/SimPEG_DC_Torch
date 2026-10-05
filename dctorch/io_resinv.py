"""
ResInv .dat pipeline for the MAR campaign (port of _gpubench/mar_pipeline.py
to stock simpeg >=0.25 — only the imports changed).

load_line(dat)            -> dict with name, ABMN x-coords, RES, topo interpolator
build_survey(L)           -> fresh Survey + dobs aligned (grouped by source)
build_mesh(dh, L, survey) -> TreeMesh + active_cells (electrodes draped in place)
"""
import csv as csvmod
import os
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d

from simpeg.electromagnetics.static import resistivity as dc
from discretize import TreeMesh
from discretize.utils import active_from_xyz

# The campaign topography. Set DCTORCH_TOPO_CSV to point this somewhere else;
# the default is where it sits on the machine these benchmarks were run on.
_TOPO_WIN = os.environ.get(
    "DCTORCH_TOPO_CSV_WIN",
    r"C:\campaign\raw_data\Topography\topografia.csv")
# The same drive is reached by two paths depending on which side of WSL the
# interpreter is on, and benchmarks that time two engines have to run on both.
TOPO_CSV = os.environ.get("DCTORCH_TOPO_CSV") or (
    _TOPO_WIN if os.path.exists(_TOPO_WIN)
    else _TOPO_WIN.replace("\\", "/").replace("C:", "/mnt/c"))


def load_line(dat_path, line=None, topo_csv=TOPO_CSV, qc_mad=4.0):
    """Parse a ResInv .dat, apply log-MAD QC, and attach the line topography."""
    rows = []
    with open(dat_path) as f:
        lines = f.read().splitlines()
    name = lines[0].strip()
    for ln in lines[2:]:
        t = ln.split()
        if len(t) >= 10 and t[0] == "4":
            rows.append([float(x) for x in t[1:11]])
    D = np.array(rows)
    xa, za, xb, zb, xm, zm, xn, zn, RES, IP = D.T
    K = 2*np.pi/(1/np.abs(xm-xa) - 1/np.abs(xn-xa)
                 - 1/np.abs(xm-xb) + 1/np.abs(xn-xb))
    rhoa = K*RES
    ok = rhoa > 0
    lr = np.log(rhoa[ok])
    med = np.median(lr)
    mad = 1.4826*np.median(np.abs(lr-med))
    ok2 = ok.copy()
    ok2[ok] = np.abs(lr-med) < qc_mad*mad
    xa, xb, xm, xn, RES = (v[ok2] for v in (xa, xb, xm, xn, RES))

    topo_line = line if line is not None else str(int(os.path.basename(dat_path)[4:7]))
    dd, zz = [], []
    with open(topo_csv) as f:
        for row in csvmod.DictReader(f):
            if row["LINE"] == topo_line:
                dd.append(float(row["D"]))
                zz.append(float(row["Z"]))
    dd = np.asarray(dd)
    zz = np.asarray(zz)
    srt = np.argsort(dd)
    dd, zz = dd[srt], zz[srt]
    fint = interp1d(dd, zz, fill_value=(zz[0], zz[-1]), bounds_error=False)
    xg = np.linspace(dd.min()-600, dd.max()+600, 601)
    topo2d = np.c_[xg, fint(np.clip(xg, dd.min(), dd.max()))]

    return dict(name=name, xa=xa, xb=xb, xm=xm, xn=xn, RES=RES,
                rho_ref=float(np.exp(np.median(np.log(rhoa[ok2])))),
                fint=fint, topo2d=topo2d, topo_line=topo_line)


def build_survey(L):
    """Fresh Survey (grouped by source) + dobs realigned. Build one per mesh:
    drape_electrodes_on_topography mutates the survey."""
    fint = L["fint"]
    groups = OrderedDict()
    for i in range(len(L["RES"])):
        groups.setdefault((L["xa"][i], L["xb"][i]), []).append(i)
    srcs, order = [], []
    for (a, b), idx in groups.items():
        idx = np.asarray(idx)
        rx = dc.receivers.Dipole(np.c_[L["xm"][idx], fint(L["xm"][idx])],
                                 np.c_[L["xn"][idx], fint(L["xn"][idx])],
                                 data_type="volt")
        srcs.append(dc.sources.Dipole([rx], np.r_[a, float(fint(a))],
                                      np.r_[b, float(fint(b))]))
        order.append(idx)
    survey = dc.survey.Survey(srcs)
    dobs = L["RES"][np.concatenate(order)]
    return survey, dobs


def build_mesh(dh, L, survey):
    """Campaign TreeMesh recipe; drapes the survey electrodes in place."""
    topo2d = L["topo2d"]
    nbx = 2**int(np.round(np.log(2400./dh)/np.log(2)))
    nbz = 2**int(np.round(np.log(1200./dh)/np.log(2)))
    mesh = TreeMesh([[(dh, nbx)], [(dh, nbz)]], x0="CN", diagonal_balance=True)
    mesh.origin = mesh.origin + np.r_[200.0, topo2d[:, 1].max()]
    mesh.refine_surface(topo2d, padding_cells_by_level=[0, 0, 4, 4],
                        finalize=False)
    mesh.refine_points(survey.unique_electrode_locations,
                       padding_cells_by_level=[8, 12, 6, 6], finalize=False)
    mesh.finalize()
    active = active_from_xyz(mesh, topo2d)
    survey.drape_electrodes_on_topography(mesh, active, option="top")
    return mesh, active
