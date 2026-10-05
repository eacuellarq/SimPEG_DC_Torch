"""J^T v against stock SimPEG's own Jtvec, and against finite differences.

Three checks per engine:
  (a) parity with `sim.Jtvec(m, v)` from stock SimPEG 0.25, which forms the
      sensitivities the conventional way;
  (b) a directional identity that needs no reference at all,
      (J^T v) . u == d/de [ v . dpred(m + e u) ],  by central differences;
  (c) 2.5-D only: the chunked product equals the whole-survey one, so source
      batching does not change the answer.

Runs on the GPU when there is one and on the cpu path otherwise.

  python -m dctorch.tests.test_jtvec
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64


def main():
    from dctorch.tests.test_suite import make_case, stock_sim, engine

    rng = np.random.default_rng(19)
    for kind in ("2d-tensor", "2d-tree", "3d-tensor", "3d-tree"):
        mesh, survey, m = make_case(kind)
        sim = stock_sim(kind, mesh, survey)
        eng = engine(kind, mesh, survey, DEV)
        mt = torch.tensor(m, dtype=DT, device=DEV)
        v = rng.standard_normal(survey.nD)
        vt = torch.tensor(v, dtype=DT, device=DEV)

        # (a) against the sensitivities stock SimPEG builds itself
        g_ref = sim.Jtvec(m, v)
        g = eng.Jtvec(mt, vt).cpu().numpy()
        rel = np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref)

        # (b) the directional identity, no reference needed
        u = rng.standard_normal(m.size)
        u /= np.linalg.norm(u)
        eps = 1e-5

        def vd(mm):
            s = stock_sim(kind, mesh, survey)      # fresh: the model setter
            return float(v @ s.dpred(mm))          # ignores small perturbations

        fd = (vd(m + eps * u) - vd(m - eps * u)) / (2 * eps)
        an = float(g @ u)
        relfd = abs(fd - an) / (abs(fd) + 1e-300)

        line = (f"[{kind:9s}] nD {survey.nD:4d} nM {mesh.nC:6,} | "
                f"vs stock Jtvec {rel:.2e} | vs central differences {relfd:.2e}")

        # (c) chunking must not change it
        if kind.startswith("2d"):
            nS = len(survey.source_list)
            g_ch = eng.Jtvec(mt, vt, chunk=max(1, nS // 3)).cpu().numpy()
            relc = np.linalg.norm(g_ch - g) / np.linalg.norm(g)
            line += f" | chunked vs whole {relc:.2e}"
            assert relc < 1e-12, f"chunking changed J^T v in {kind}: {relc}"

        print(line)
        assert rel < 1e-9, f"J^T v disagrees with stock in {kind}: {rel}"
        assert relfd < 1e-6, f"J^T v fails the FD identity in {kind}: {relfd}"

    print(f"JTVEC VALIDADO ({DEV})")


if __name__ == "__main__":
    main()
