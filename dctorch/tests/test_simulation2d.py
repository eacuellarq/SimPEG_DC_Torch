"""
Validación Fase 2: TorchDC2D vs SimPEG 0.25 stock (dpred) y vs FD (gradiente).
Correr desde la raíz del repo:  python -m dctorch.tests.test_simulation2d
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time

import numpy as np
import torch

# Runs on the GPU when there is one and on the cpu path otherwise: the two
# asserts here are parity against stock SimPEG and the gradient against finite
# differences, neither of which depends on the device. Only the timing printed
# at the end does.
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def build_problem():
    from discretize import TensorMesh
    from simpeg import maps
    from simpeg.electromagnetics.static import resistivity as dc

    mesh = TensorMesh([[(1.0, 100)], [(1.0, 50)]], origin="CN")
    xe = np.linspace(-30, 30, 21)
    srcs = []
    for i in range(len(xe) - 1):        # dipolo-dipolo n=1..6 (con profundidad)
        lm, ln = [], []
        for n in range(1, 7):
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
    survey = dc.survey.Survey(srcs)

    # modelo: fondo 100 ohm-m + bloque conductor
    m = np.log(100.0) * np.ones(mesh.nC)
    cc = mesh.cell_centers
    blk = (np.abs(cc[:, 0] - 5) < 8) & (cc[:, 1] < -3) & (cc[:, 1] > -10)
    m[blk] = np.log(10.0)

    from pymatsolver import Pardiso
    sim = dc.simulation_2d.Simulation2DNodal(
        mesh, survey=survey, rhoMap=maps.ExpMap(mesh), nky=11, solver=Pardiso)
    return mesh, survey, sim, m


def main():
    from dctorch import TorchDC2D

    mesh, survey, sim, m = build_problem()
    print(f"mesh {mesh.shape_cells} nC={mesh.nC:,} nN={mesh.n_nodes:,} | "
          f"nD={survey.nD} | nky=11")

    t0 = time.perf_counter()
    eng = TorchDC2D(mesh, survey, nky=11, device=DEV, quadrature="simpeg")
    print(f"[build] extraccion + plan: {time.perf_counter()-t0:.1f}s "
          f"(una vez por malla/survey)")

    # ---- paridad dpred vs stock 0.25 (Pardiso CPU) ----
    t0 = time.perf_counter()
    d_ref = sim.dpred(m)
    t_ref = time.perf_counter() - t0
    mt = torch.tensor(m, device=DEV)
    d_t = eng.dpred(mt).cpu().numpy()
    rel = np.linalg.norm(d_t - d_ref) / np.linalg.norm(d_ref)
    print(f"[paridad] dpred torch vs stock: rel-err {rel:.2e} "
          f"(stock Pardiso: {t_ref:.2f}s)")
    assert rel < 1e-10

    # ---- gradiente vs FD (direccional, 3 direcciones) ----
    rng = np.random.default_rng(3)
    dobs = torch.tensor(d_ref * (1 + 0.03 * rng.standard_normal(len(d_ref))),
                        device=DEV)
    w = torch.tensor(1.0 / (0.03 * np.abs(d_ref) + 1e-12), device=DEV)

    mt = torch.tensor(m, device=DEV, requires_grad=True)
    phi = eng.misfit(mt, dobs, w)
    phi.backward()
    g = mt.grad.cpu().numpy()

    def phi_np(mm):
        # sim FRESCO por evaluacion: el setter de model de SimPEG ignora
        # perturbaciones bajo su umbral allclose (leccion vieja, vigente en 0.25)
        from simpeg import maps
        from simpeg.electromagnetics.static import resistivity as dc
        from pymatsolver import Pardiso
        s = dc.simulation_2d.Simulation2DNodal(
            mesh, survey=survey, rhoMap=maps.ExpMap(mesh), nky=11,
            solver=Pardiso)
        r = w.cpu().numpy() * (s.dpred(mm) - dobs.cpu().numpy())
        return float((r ** 2).sum())

    eps, rels = 1e-4, []
    for _ in range(3):
        v = rng.standard_normal(len(m))
        v /= np.linalg.norm(v)
        fd = (phi_np(m + eps * v) - phi_np(m - eps * v)) / (2 * eps)
        an = float(g @ v)
        rels.append(abs(fd - an) / (abs(fd) + 1e-300))
    print(f"[gradiente] direccional vs FD (stock forward): "
          f"max rel {max(rels):.2e}")
    assert max(rels) < 1e-6

    # ---- timing ----
    def run_eval():
        d = eng.dpred(mt.detach())
        _sync()
        return d

    def run_grad():
        mt.grad = None
        eng.misfit(mt, dobs, w).backward()
        _sync()

    run_eval(); run_grad()
    t0 = time.perf_counter()
    for _ in range(10):
        run_eval()
    t_ev = 100 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for _ in range(10):
        run_grad()
    t_gr = 100 * (time.perf_counter() - t0)
    print(f"[tiempo] eval {t_ev:.1f} ms | eval+grad {t_gr:.1f} ms | "
          f"stock CPU eval {1e3*t_ref:.0f} ms ({t_ref*1e3/t_ev:.1f}x)")
    print("FASE 2 VALIDADA")


if __name__ == "__main__":
    main()
