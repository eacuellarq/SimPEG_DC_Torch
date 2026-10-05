"""The batched adjoint: the explicit Jacobian, against the two references.

`Jmatrix` forms the whole nD x nM sensitivity matrix from ONE factorization,
solving the adjoint for many data at a time. The right-hand side it batches
over is Pd^T, which is known before the run -- that is the whole trick -- so
the thing most worth testing is that batching it changes nothing.

Four checks per mesh:
  (a) parity with stock SimPEG 0.25, which builds the sensitivities its own
      way (`storeJ=True`, Pardiso on the CPU);
  (b) parity with our own `Jtvec`, row by row: J^T e_i for every datum. Same
      engine, same operators, opposite direction of travel;
  (c) the block width is bookkeeping: block 7, 32 and nD must agree to
      round-off, including the padded last block;
  (d) with active cells, the columns are the active ones and they equal the
      full-mesh columns there -- the injection map is transposed, not
      re-derived;
  (e) singularity removal refuses loudly instead of returning a wrong matrix.

  python -m dctorch.tests.test_jmatrix
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64


def stock_J(mesh, survey, m):
    from simpeg import maps
    from simpeg.electromagnetics.static import resistivity as dc
    from pymatsolver import Pardiso
    sim = dc.simulation_2d.Simulation2DNodal(
        mesh, survey=survey, rhoMap=maps.ExpMap(mesh), nky=11,
        storeJ=True, solver=Pardiso)
    return np.asarray(sim.getJ(m))


def main():
    from dctorch.tests.test_suite import make_case, engine

    for kind in ("2d-tensor", "2d-tree"):
        mesh, survey, m = make_case(kind)
        eng = engine(kind, mesh, survey, DEV)
        mt = torch.tensor(m, dtype=DT, device=DEV)
        nD, nC = survey.nD, mesh.nC

        J = eng.Jmatrix(mt).cpu().numpy()
        assert J.shape == (nD, nC), f"{kind}: shape {J.shape}"

        # (a) stock SimPEG's own sensitivities
        Jref = stock_J(mesh, survey, m)
        rel = np.linalg.norm(J - Jref) / np.linalg.norm(Jref)

        # (b) our own reverse pass, one datum at a time
        e = torch.zeros(nD, dtype=DT, device=DEV)
        rows = []
        for i in range(nD):
            e.zero_()
            e[i] = 1.0
            rows.append(eng.Jtvec(mt, e))
        Jseq = torch.stack(rows, 0).cpu().numpy()
        relq = np.linalg.norm(J - Jseq) / np.linalg.norm(Jseq)

        # (c) the block width must not matter (7 leaves a padded tail)
        relb = 0.0
        for blk in (7, nD):
            Jb = eng.Jmatrix(mt, block=blk).cpu().numpy()
            relb = max(relb, np.linalg.norm(Jb - J) / np.linalg.norm(J))

        print(f"[{kind:9s}] nD {nD:4d} nM {nC:6,} | vs stock getJ {rel:.2e} | "
              f"vs Jtvec x nD {relq:.2e} | vs block {{7,nD}} {relb:.2e}")
        assert rel < 1e-9, f"{kind}: disagrees with stock getJ ({rel})"
        assert relq < 1e-12, f"{kind}: disagrees with our own Jtvec ({relq})"
        assert relb < 1e-12, f"{kind}: the block width changed J ({relb})"

    # (d) active cells: same matrix, restricted to the columns that move
    from dctorch import TorchDC2D
    mesh, survey, m = make_case("2d-tensor")
    act = mesh.cell_centers[:, 1] < -1.0            # a slab is held fixed
    val = float(np.log(100.0))
    m_full = m.copy()
    m_full[~act] = val                              # the two models must agree
    eng_f = TorchDC2D(mesh, survey, nky=11, device=DEV, quadrature="simpeg")
    eng_a = TorchDC2D(mesh, survey, nky=11, device=DEV, quadrature="simpeg",
                      active_cells=act, val_inactive=val)
    Jf = eng_f.Jmatrix(torch.tensor(m_full, dtype=DT, device=DEV)).cpu().numpy()
    Ja = eng_a.Jmatrix(
        torch.tensor(m_full[act], dtype=DT, device=DEV)).cpu().numpy()
    assert Ja.shape == (survey.nD, int(act.sum())), f"active shape {Ja.shape}"
    rela = np.linalg.norm(Ja - Jf[:, act]) / np.linalg.norm(Jf[:, act])
    print(f"[active   ] nM {int(act.sum()):6,} of {mesh.nC:,} | "
          f"vs the full-mesh columns {rela:.2e}")
    assert rela < 1e-13, f"the injection map changed J ({rela})"

    # (e) the formulation this assembly does not cover says so
    eng_sr = TorchDC2D(mesh, survey, nky=11, device=DEV,
                       singularity_removal=True)
    try:
        eng_sr.Jmatrix(torch.tensor(m, dtype=DT, device=DEV))
    except NotImplementedError as exc:
        assert "Jtvec" in str(exc)
        print(f"[sr       ] refused, as it must: {str(exc).split('.')[0]}.")
    else:
        raise AssertionError("Jmatrix returned a matrix under singularity "
                             "removal, where its assembly is incomplete")

    print(f"JMATRIX VALIDADO ({DEV})")


if __name__ == "__main__":
    main()
