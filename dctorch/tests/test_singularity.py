"""Singularity removal: accuracy against the analytic half space, and the property
that must survive it -- an exact gradient.

Run:  python -m dctorch.tests.test_singularity
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg.electromagnetics.static import resistivity as dc

from ..simulation2d import TorchDC2D

RHO, NKY, DH = 200.0, 21, 2.5


def build():
    x = np.arange(0.0, 121.0, 5.0)
    elec = np.c_[x, np.zeros(x.size)]
    abmn = []
    for i in range(x.size - 3):
        for n in range(1, 7):
            if i + 1 + n + 1 < x.size:
                abmn.append((i, i + 1, i + 1 + n, i + 2 + n))
    abmn = np.array(abmn)
    src, order = [], []
    for (a, b) in dict.fromkeys(map(tuple, abmn[:, :2])):
        k = np.where((abmn[:, 0] == a) & (abmn[:, 1] == b))[0]
        src.append(dc.sources.Dipole(
            [dc.receivers.Dipole(locations_m=elec[abmn[k, 2]], locations_n=elec[abmn[k, 3]])],
            location_a=elec[a], location_b=elec[b]))
        order.append(k)
    survey = dc.Survey(src)
    perm = np.concatenate(order)

    n = 0
    while sum(DH * 1.35 ** np.arange(1, n + 1)) < 1000.0:
        n += 1
    pad = sum(DH * 1.35 ** np.arange(1, n + 1))
    mesh = TensorMesh([[(DH, n, -1.35), (DH, int(x.max() / DH)), (DH, n, 1.35)],
                       [(DH, n, -1.35), (DH, int(50.0 / DH))]], origin=[-pad, -50.0 - pad])
    act = active_from_xyz(mesh, np.c_[np.r_[-1e3, x, 1e3], np.zeros(x.size + 2)])
    survey.drape_electrodes_on_topography(mesh, act, topo_cell_cutoff="top",
                                          shift_horizontal=False)

    r = lambda i, j: np.linalg.norm(elec[i] - elec[j])
    truth = np.array([RHO * (1 / r(a, m) - 1 / r(b, m) - 1 / r(a, n_) + 1 / r(b, n_))
                      / (2 * np.pi) for a, b, m, n_ in abmn])[perm]
    return mesh, act, survey, truth


def main():
    mesh, act, survey, truth = build()
    nC = int(act.sum())
    m = torch.full((nC,), np.log(RHO), dtype=torch.float64, device="cuda")
    err = {}
    for sr in (False, True):
        eng = TorchDC2D(mesh, survey, nky=NKY, device="cuda", dtype=torch.float64,
                        active_cells=act, val_inactive=np.log(1e8),
                        singularity_removal=sr, rho_primary=RHO)
        with torch.no_grad():
            V = eng.dpred(m).cpu().numpy()
        err[sr] = 100 * np.abs(V / truth - 1)
        print(f"[{'secondary' if sr else 'total':>9}] {mesh.n_cells:,} cells | "
              f"median {np.median(err[sr]):.3f} % | max {err[sr].max():.3f} %")

        # the gradient must stay exact: directional AD against a central difference
        rng = np.random.default_rng(0)
        m0 = torch.tensor(np.log(RHO) + 0.3 * rng.standard_normal(nC),
                          dtype=torch.float64, device="cuda")
        dobs = torch.tensor(truth, dtype=torch.float64, device="cuda")
        w = 1.0 / (0.05 * dobs.abs())
        mg = m0.clone().requires_grad_(True)
        f = eng.misfit(mg, dobs, w)
        f.backward()
        v = torch.tensor(rng.standard_normal(nC), dtype=torch.float64, device="cuda")
        v /= v.norm()
        h = 1e-5
        with torch.no_grad():
            fd = float((eng.misfit(m0 + h * v, dobs, w) - eng.misfit(m0 - h * v, dobs, w)) / (2 * h))
        ad = float(mg.grad @ v)
        rel = abs(ad - fd) / abs(fd)
        print(f"{'':11} gradient vs finite difference: rel {rel:.2e}")
        assert rel < 1e-6, rel

        # chunking must not move the answer
        eng.chunk = 4
        with torch.no_grad():
            Vc = eng.dpred(m).cpu().numpy()
        assert np.abs(Vc / V - 1).max() < 1e-12, np.abs(Vc / V - 1).max()
        del eng
        torch.cuda.empty_cache()

    gain = np.median(err[False]) / np.median(err[True])
    print(f"the split is {gain:.0f}x more accurate on this mesh")
    assert np.median(err[True]) < 0.2, np.median(err[True])
    assert gain > 5, gain
    print("SINGULARITY REMOVAL VALIDATED")


if __name__ == "__main__":
    main()
