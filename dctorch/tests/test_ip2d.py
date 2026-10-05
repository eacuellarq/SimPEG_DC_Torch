"""
Validación IP linealizado: TorchIP2D vs induced_polarization.Simulation2DNodal
stock 0.25 (dpred y J^T v) + FD del operador lineal, en cuda y cpu.
Correr desde la raíz del repo:  python -m dctorch.tests.test_ip2d
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


def build_problem():
    from discretize import TensorMesh
    from simpeg import maps
    from simpeg.electromagnetics.static import induced_polarization as ip
    from simpeg.electromagnetics.static import resistivity as dc

    mesh = TensorMesh([[(1.0, 100)], [(1.0, 50)]], origin="CN")
    xe = np.linspace(-30, 30, 21)
    srcs = []
    for i in range(len(xe) - 1):        # dipolo-dipolo n=1..6
        lm, ln = [], []
        for n in range(1, 7):
            jm, jn = i + 1 + n, i + 2 + n
            if jn >= len(xe):
                break
            lm.append([xe[jm], 0.0])
            ln.append([xe[jn], 0.0])
        if lm:
            rx = dc.receivers.Dipole(np.array(lm), np.array(ln),
                                     data_type="apparent_chargeability")
            srcs.append(dc.sources.Dipole([rx], np.r_[xe[i], 0.0],
                                          np.r_[xe[i + 1], 0.0]))
    survey = dc.survey.Survey(srcs)

    # fondo: 100 ohm-m + bloque conductor | eta: bloque cargable desplazado
    m_dc = np.log(100.0) * np.ones(mesh.nC)
    cc = mesh.cell_centers
    blk = (np.abs(cc[:, 0] - 5) < 8) & (cc[:, 1] < -3) & (cc[:, 1] > -10)
    m_dc[blk] = np.log(10.0)
    eta = np.zeros(mesh.nC)
    blk2 = (np.abs(cc[:, 0] + 6) < 6) & (cc[:, 1] < -2) & (cc[:, 1] > -8)
    eta[blk2] = 0.1

    from pymatsolver import Pardiso
    sim = ip.Simulation2DNodal(
        mesh, survey=survey, sigma=np.exp(-m_dc),
        etaMap=maps.IdentityMap(mesh), nky=11, solver=Pardiso)
    return mesh, survey, sim, m_dc, eta


def main():
    from dctorch import TorchIP2D

    mesh, survey, sim, m_dc, eta = build_problem()
    print(f"mesh {mesh.shape_cells} nC={mesh.nC:,} | nD={survey.nD} | nky=11")

    d_ref = np.asarray(sim.dpred(eta))
    rng = np.random.default_rng(3)
    v = rng.standard_normal(survey.nD)
    g_ref = np.asarray(sim.Jtvec(eta, v))

    for device in ("cuda", "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            print("[cuda] no disponible, salto")
            continue
        eng = TorchIP2D(mesh, survey, m_dc, nky=11, device=device, quadrature="simpeg")
        et = torch.tensor(eta, device=device)
        d_t = eng.dpred(et).cpu().numpy()
        rel = np.linalg.norm(d_t - d_ref) / np.linalg.norm(d_ref)
        print(f"[{device}] paridad dpred IP vs stock: rel {rel:.2e}")
        assert rel < 1e-10

        et = torch.tensor(eta, device=device, requires_grad=True)
        vt = torch.tensor(v, device=device)
        (eng.dpred(et) * vt).sum().backward()
        g_t = et.grad.cpu().numpy()
        relg = np.linalg.norm(g_t - g_ref) / np.linalg.norm(g_ref)
        print(f"[{device}] paridad J^T v vs stock Jtvec: rel {relg:.2e}")
        assert relg < 1e-10

        # FD direccional del misfit (operador lineal => casi exacto)
        dobs = torch.tensor(d_ref * (1 + 0.05 * rng.standard_normal(
            len(d_ref))), device=device)
        w = torch.tensor(1.0 / (0.05 * np.abs(d_ref) + 1e-6), device=device)
        p = torch.tensor(rng.standard_normal(mesh.nC), device=device)
        et = torch.tensor(eta, device=device, requires_grad=True)
        phi = eng.misfit(et, dobs, w)
        phi.backward()
        gp = float((et.grad * p).sum())
        eps = 1e-4
        with torch.no_grad():
            fp = float(eng.misfit(et + eps * p, dobs, w))
            fm = float(eng.misfit(et - eps * p, dobs, w))
        fd = (fp - fm) / (2 * eps)
        err = abs(fd - gp) / abs(fd)
        print(f"[{device}] FD misfit: rel {err:.2e}")
        assert err < 1e-7

    print("test_ip2d OK")


if __name__ == "__main__":
    main()
