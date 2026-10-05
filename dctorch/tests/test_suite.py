"""
Suite de cobertura dctorch: {2.5D, 3D} x {TensorMesh, TreeMesh} x {cuda, cpu}.
Cada combinacion: paridad dpred vs SimPEG 0.25 stock (Pardiso) + tiempo;
gradiente direccional vs FD (sims frescos) en las combinaciones cuda.
Nota: el L_ky/L3 se AUTO-VALIDA a 1e-10 contra getA en CADA construccion
(asserts del extractor), en cualquier malla — la paridad de aqui es encima.

Correr desde la raiz del repo:  python -m dctorch.tests.test_suite
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time

import numpy as np
import torch

# The devices to sweep. Without a GPU only the cpu path exists, and the
# gradient check -- which is device-independent physics -- runs there instead of
# being skipped, so a CPU-only machine still validates every combination.
DEVICES = ("cuda", "cpu") if torch.cuda.is_available() else ("cpu",)
GDEV = DEVICES[0]


def surveys_2d(xe):
    from simpeg.electromagnetics.static import resistivity as dc
    srcs = []
    for i in range(len(xe) - 1):
        lm, ln = [], []
        for n in range(1, 5):
            jm, jn = i + 1 + n, i + 2 + n
            if jn >= len(xe):
                break
            lm.append([xe[jm], 0.0])
            ln.append([xe[jn], 0.0])
        if lm:
            rx = dc.receivers.Dipole(np.array(lm), np.array(ln),
                                     data_type="volt")
            srcs.append(dc.sources.Dipole([rx], np.r_[xe[i], 0.0],
                                          np.r_[xe[i + 1], 0.0]))
    return dc.survey.Survey(srcs)


def surveys_3d(xe):
    from simpeg.electromagnetics.static import resistivity as dc
    srcs = []
    for i in range(len(xe) - 1):
        lm, ln = [], []
        for n in range(1, 4):
            jm, jn = i + 1 + n, i + 2 + n
            if jn >= len(xe):
                break
            lm.append([xe[jm], 0.0, 0.0])
            ln.append([xe[jn], 0.0, 0.0])
        if lm:
            rx = dc.receivers.Dipole(np.array(lm), np.array(ln),
                                     data_type="volt")
            srcs.append(dc.sources.Dipole([rx], np.r_[xe[i], 0.0, 0.0],
                                          np.r_[xe[i + 1], 0.0, 0.0]))
    return dc.survey.Survey(srcs)


def make_case(kind):
    from discretize import TensorMesh, TreeMesh
    rng = np.random.default_rng(11)
    if kind == "2d-tensor":
        mesh = TensorMesh([[(1.0, 64)], [(1.0, 32)]], origin="CN")
        survey = surveys_2d(np.linspace(-20, 20, 15))
    elif kind == "2d-tree":
        mesh = TreeMesh([[(1.0, 64)], [(1.0, 64)]], x0=[-32, -64])
        pts = np.c_[np.linspace(-20, 20, 15), np.zeros(15)]
        mesh.refine_points(pts, padding_cells_by_level=[6, 6, 4], finalize=True)
        survey = surveys_2d(np.linspace(-20, 20, 15))
    elif kind == "3d-tensor":
        mesh = TensorMesh([[(2.0, 20)], [(2.0, 20)], [(2.0, 10)]],
                          origin=["C", "C", "N"])
        mesh.origin = mesh.origin + np.r_[0, 0, -mesh.h[2].sum()] * 0  # top z=0
        mesh.origin = np.r_[mesh.origin[0], mesh.origin[1], -mesh.h[2].sum()]
        survey = surveys_3d(np.linspace(-14, 14, 9))
    elif kind == "3d-tree":
        mesh = TreeMesh([[(2.0, 32)], [(2.0, 32)], [(2.0, 16)]],
                        x0=[-32, -32, -32])
        pts = np.c_[np.linspace(-14, 14, 9), np.zeros(9), np.zeros(9)]
        mesh.refine_points(pts, padding_cells_by_level=[4, 4, 2], finalize=True)
        survey = surveys_3d(np.linspace(-14, 14, 9))
    m = np.log(100.0) * np.ones(mesh.nC) \
        + 0.3 * rng.standard_normal(mesh.nC)          # modelo rugoso (no trivial)
    return mesh, survey, m


def stock_sim(kind, mesh, survey):
    from simpeg import maps
    from simpeg.electromagnetics.static import resistivity as dc
    from pymatsolver import Pardiso
    if kind.startswith("2d"):
        return dc.simulation_2d.Simulation2DNodal(
            mesh, survey=survey, rhoMap=maps.ExpMap(mesh), nky=11,
            solver=Pardiso)
    return dc.simulation.Simulation3DNodal(
        mesh, survey=survey, rhoMap=maps.ExpMap(mesh), solver=Pardiso)


def engine(kind, mesh, survey, device):
    from dctorch import TorchDC2D
    from dctorch.simulation3d import TorchDC3D
    if kind.startswith("2d"):
        return TorchDC2D(mesh, survey, nky=11, device=device, quadrature="simpeg")
    return TorchDC3D(mesh, survey, device=device)


def main():
    results = []
    for kind in ("2d-tensor", "2d-tree", "3d-tensor", "3d-tree"):
        mesh, survey, m = make_case(kind)
        sim = stock_sim(kind, mesh, survey)
        t0 = time.perf_counter()
        d_ref = sim.dpred(m)
        t_ref = time.perf_counter() - t0
        for device in DEVICES:
            eng = engine(kind, mesh, survey, device)
            mt = torch.tensor(m, device=device)
            d = eng.dpred(mt)                      # warm (build caches)
            torch.cuda.synchronize() if device == "cuda" else None
            t0 = time.perf_counter()
            d = eng.dpred(mt)
            torch.cuda.synchronize() if device == "cuda" else None
            t_ev = time.perf_counter() - t0
            rel = float(np.linalg.norm(d.cpu().numpy() - d_ref)
                        / np.linalg.norm(d_ref))
            print(f"[{kind:9s}|{device:4s}] nC {mesh.nC:6,} nD {survey.nD:4d} "
                  f"| paridad {rel:.2e} | eval {1e3*t_ev:7.1f} ms "
                  f"(stock CPU {1e3*t_ref:6.0f} ms)")
            assert rel < 1e-9, f"paridad rota en {kind}/{device}: {rel}"
            results.append((kind, device, rel))

        # gradiente direccional vs FD (sims frescos), en GDEV
        eng = engine(kind, mesh, survey, GDEV)
        rng = np.random.default_rng(5)
        dobs = torch.tensor(d_ref * (1 + 0.02 * rng.standard_normal(len(d_ref))),
                            device=GDEV)
        w = torch.tensor(1.0 / (0.02 * np.abs(d_ref) + 1e-12), device=GDEV)
        mt = torch.tensor(m, device=GDEV, requires_grad=True)
        eng.misfit(mt, dobs, w).backward()
        g = mt.grad.cpu().numpy()

        def phi(mm):
            s = stock_sim(kind, mesh, survey)      # FRESCO (setter allclose)
            r = w.cpu().numpy() * (s.dpred(mm) - dobs.cpu().numpy())
            return float((r ** 2).sum())

        eps = 1e-3
        v = rng.standard_normal(len(m))
        v /= np.linalg.norm(v)
        fd = (phi(m + eps * v) - phi(m - eps * v)) / (2 * eps)
        an = float(g @ v)
        relg = abs(fd - an) / (abs(fd) + 1e-300)
        print(f"[{kind:9s}|grad] FD vs autograd: rel {relg:.2e}")
        assert relg < 2e-4, f"gradiente roto en {kind}: {relg}"

    print("SUITE COMPLETA: {2.5D,3D} x {Tensor,Tree} x "
          + "+".join(DEVICES) + " VALIDADA")


if __name__ == "__main__":
    main()
