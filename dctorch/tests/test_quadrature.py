"""The wavenumber quadrature: the fit itself, the engine's choice of it, and
what it does to the data against the analytic half space.

  (a) the electrode fit's transform error never grows with nky (continuation),
      its weights are all positive (no amplification of the 2-D solves'
      errors), and it succeeds where SimPEG's fit falls back (nky = 31);
  (b) quadrature="auto" at a healthy nky keeps SimPEG's points: dpred matches a
      stock simulation to round-off, as before;
  (c) quadrature="auto" at nky = 31 detects SimPEG's trapezoid fallback,
      replaces it, and says so;
  (d) end to end, on the singularity-removal half space (so the mesh error at
      the source is gone and the transform is what is left): the electrode fit
      repairs nky = 31 and is at least as accurate as SimPEG's at every nky.

  python -m dctorch.tests.test_quadrature
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fit_checks():
    from dctorch.quadrature import k0_quadrature
    _, _, info = k0_quadrature(5.0, 120.0, 31)
    eb = info["err_by_n"]
    ns = sorted(eb)
    worst_rise = max(eb[b] / eb[a] for a, b in zip(ns, ns[1:]))
    print(f"[fit] 5-120 m | err nky 7 {eb[7]:.1e}, 11 {eb[11]:.1e}, 21 {eb[21]:.1e}, "
          f"31 {eb[31]:.1e} | worst step-to-step rise x{worst_rise:.2f}")
    print(f"[fit] nky 31 -> {info['n_used']} used | amplification sum|w|/sum w "
          f"{info['amplification']:.4f}")
    assert info["success"], "electrode fit reported failure at nky=31"
    assert info["amplification"] < 1.0 + 1e-12, "weights are not all positive"
    assert worst_rise < 1.5, f"error grew with nky (x{worst_rise:.2f})"
    assert eb[31] < 1e-5, eb[31]


def auto_checks():
    from simpeg import maps
    from simpeg.electromagnetics.static import resistivity as dc
    from pymatsolver import Pardiso
    from dctorch import TorchDC2D
    from dctorch.tests.test_suite import make_case

    mesh, survey, m = make_case("2d-tensor")
    eng = TorchDC2D(mesh, survey, nky=11, device=DEV, quadrature="auto")
    assert eng.quadrature_info["source"] == "simpeg"
    sim = dc.simulation_2d.Simulation2DNodal(mesh, survey=survey,
                                             rhoMap=maps.ExpMap(mesh), nky=11,
                                             solver=Pardiso)
    d_ref = sim.dpred(m)
    with torch.no_grad():
        d = eng.dpred(torch.tensor(m, device=DEV)).cpu().numpy()
    par = np.linalg.norm(d - d_ref) / np.linalg.norm(d_ref)
    print(f"[auto nky=11] keeps SimPEG's points | parity with stock {par:.1e}")
    assert par < 1e-10, par

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng31 = TorchDC2D(mesh, survey, nky=31, device=DEV, quadrature="auto")
    qi = eng31.quadrature_info
    said = any("fell back" in str(w.message) for w in caught)
    print(f"[auto nky=31] SimPEG fell back: {qi['simpeg_fell_back']} | replaced: "
          f"{qi['source'] == 'electrodes'} | warned: {said} | transform error "
          f"{qi.get('simpeg_max_rel_err', float('nan')):.1e} -> {qi['max_rel_err']:.1e}")
    assert qi["simpeg_fell_back"] and qi["source"] == "electrodes" and said


def halfspace_checks():
    from dctorch import TorchDC2D
    from dctorch.tests.test_singularity import RHO, build

    mesh, act, survey, truth = build()
    m = torch.full((int(act.sum()),), np.log(RHO), dtype=torch.float64, device=DEV)
    res = {}
    print(f"[half space] {mesh.n_cells:,} cells, singularity removal on | "
          f"median / max error vs analytic, %")
    for nky in (7, 11, 21, 31):
        row = []
        for q in ("simpeg", "electrodes"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eng = TorchDC2D(mesh, survey, nky=nky, device=DEV, active_cells=act,
                                val_inactive=np.log(1e8), singularity_removal=True,
                                rho_primary=RHO, quadrature=q)
            with torch.no_grad():
                V = eng.dpred(m).cpu().numpy()
            e = 100 * np.abs(V / truth - 1)
            res[(nky, q)] = (np.median(e), e.max())
            row.append(f"{q:>10} {np.median(e):6.3f} / {e.max():6.3f}")
            del eng
        print(f"  nky {nky:2d} | " + " | ".join(row))
    assert res[(31, "simpeg")][0] > 1.0, "expected SimPEG's fallback to show at nky=31"
    assert res[(31, "electrodes")][0] < 0.2, res[(31, "electrodes")]
    for nky in (7, 11, 21, 31):
        assert res[(nky, "electrodes")][0] <= 1.05 * res[(nky, "simpeg")][0] + 0.01, \
            f"electrode fit worse than SimPEG's at nky={nky}"


def main():
    fit_checks()
    auto_checks()
    halfspace_checks()
    print(f"QUADRATURE VALIDATED ({DEV})")


if __name__ == "__main__":
    main()
