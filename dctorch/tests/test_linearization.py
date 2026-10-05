"""The linearization Gauss-Newton runs on: J v, J^T v and diag(J^T D J) at
one model, against the explicit Jacobian.

`Jmatrix` is already validated against stock SimPEG's `getJ` and against
`Jtvec` row by row (test_jmatrix), so it is the reference here. Checks per
mesh, on the cpu path and on the gpu when there is one:

  (a) dpred of the linearization equals the engine's own dpred;
  (b) J v equals Jmatrix @ v, and J^T w equals Jmatrix^T @ w;
  (c) the adjoint identity <J v, w> = <v, J^T w> to round-off;
  (d) diag(J^T D J) equals the column sums of D J^2, and the streamed J of
      the linearization equals Jmatrix;
  (e) source chunking and dropping the stored fields change nothing;
  (f) a factorization replaced behind its back (a line-search dpred at
      another model) is detected and restored;
  (g) active cells: the products live on the active columns.

  python -m dctorch.tests.test_linearization
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DT = torch.float64


def rel(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


def check(eng, m, tag):
    dev = eng.device
    nD = eng.nD
    J = eng.Jmatrix(m)
    nM = J.shape[1]
    g = torch.Generator(device="cpu").manual_seed(3)
    v = torch.randn(nM, generator=g, dtype=DT).to(dev)
    w = torch.randn(nD, generator=g, dtype=DT).to(dev)
    D = torch.rand(nD, generator=g, dtype=DT).to(dev) + 0.1

    lin = eng.linearize(m)
    e_d = rel(lin.d, eng.dpred(m))                                   # (a)
    e_jv = rel(lin.Jvec(v), J @ v)                                   # (b)
    e_jt = rel(lin.Jtvec(w), J.T @ w)
    adj = abs(float(lin.Jvec(v) @ w - v @ lin.Jtvec(w))) \
        / float(torch.linalg.norm(J @ v) * torch.linalg.norm(w))     # (c)
    e_dg = rel(lin.diag_JtDJ(D), (D[:, None] * J ** 2).sum(0))       # (d)

    e_J = rel(lin.Jmatrix(), J)                                      # (d')
    lin_nf = eng.linearize(m, store_fields=False)                    # (e)
    e_nf = max(rel(lin_nf.Jvec(v), J @ v), rel(lin_nf.Jtvec(w), J.T @ w))

    eng.dpred(m + 0.3)                     # (f) clobbers the shared factorization
    e_cl = rel(lin.Jvec(v), J @ v)

    print(f"[{tag:16s}] nD {nD:4d} nM {nM:6,} | dpred {e_d:.1e} | Jv {e_jv:.1e} "
          f"| Jtw {e_jt:.1e} | adjoint {adj:.1e} | diag {e_dg:.1e} "
          f"| J {e_J:.1e} | no-fields {e_nf:.1e} | clobbered {e_cl:.1e}")
    assert e_d < 1e-12, f"{tag}: dpred {e_d}"
    assert e_jv < 1e-10 and e_jt < 1e-10, f"{tag}: products {e_jv} {e_jt}"
    assert adj < 1e-12, f"{tag}: adjoint identity {adj}"
    assert e_dg < 1e-10, f"{tag}: diag {e_dg}"
    assert e_J < 1e-10, f"{tag}: streamed J {e_J}"
    assert e_nf < 1e-10, f"{tag}: without stored fields {e_nf}"
    assert e_cl < 1e-10, f"{tag}: clobbered factorization not restored {e_cl}"


def main():
    from dctorch import TorchDC2D
    from dctorch.tests.test_suite import make_case

    for dev in DEVICES:
        for kind in ("2d-tensor", "2d-tree"):
            mesh, survey, m = make_case(kind)
            mt = torch.tensor(m, dtype=DT, device=dev)
            for chunk in (None, 4):
                eng = TorchDC2D(mesh, survey, nky=11, device=dev, chunk=chunk)
                check(eng, mt, f"{kind} {dev} c{chunk or 'all'}")

        # (g) active cells
        mesh, survey, m = make_case("2d-tensor")
        act = mesh.cell_centers[:, 1] < -1.0
        eng = TorchDC2D(mesh, survey, nky=11, device=dev, active_cells=act,
                        val_inactive=float(np.log(100.0)), chunk=4)
        check(eng, torch.tensor(m[act], dtype=DT, device=dev), f"active {dev}")

    print(f"LINEARIZACION VALIDADA ({', '.join(DEVICES)})")


if __name__ == "__main__":
    main()
