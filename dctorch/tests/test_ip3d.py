"""
Validación IP linealizado 3D: TorchIP3D vs induced_polarization.
Simulation3DNodal stock 0.25 (dpred y J^T v) + FD, en {tensor, tree+actives}
x {cuda, cpu}. El caso tree lleva ACTIVE CELLS (aire arriba, val_inactive
log 1e8) = el régimen de campo.
Correr desde la raíz del repo:  python -m dctorch.tests.test_ip3d
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from .test_suite import make_case


def build_ip_case(kind):
    from simpeg import maps
    from simpeg.electromagnetics.static import induced_polarization as ip
    from simpeg.electromagnetics.static import resistivity as dc
    from pymatsolver import Pardiso

    mesh, survey_dc, m_dc = make_case(kind)
    # survey IP: misma geometría, data_type apparent_chargeability
    srcs = []
    for src in survey_dc.source_list:
        rxs = [dc.receivers.Dipole(rx.locations[0], rx.locations[1],
                                   data_type="apparent_chargeability")
               for rx in src.receiver_list]
        srcs.append(dc.sources.Dipole(rxs, src.location[0], src.location[1]))
    survey = dc.survey.Survey(srcs)

    rng = np.random.default_rng(5)
    cc = mesh.cell_centers
    if kind == "3d-tree":                     # régimen de campo: aire arriba
        act = cc[:, 2] < -2.0
        m_act = m_dc[act]
        m_full = np.full(mesh.nC, np.log(1e8))
        m_full[act] = m_act
        eta_act = rng.uniform(0.0, 10.0, act.sum())
        eta_full = np.zeros(mesh.nC)
        eta_full[act] = eta_act
        etaMap = maps.InjectActiveCells(mesh, active_cells=act,
                                        value_inactive=0.0)
    else:
        act, m_act, eta_act = None, m_dc, rng.uniform(0.0, 10.0, mesh.nC)
        m_full, eta_full, etaMap = m_dc, eta_act, maps.IdentityMap(mesh)

    sim = ip.Simulation3DNodal(
        mesh, survey=survey, sigma=np.exp(-m_full), etaMap=etaMap,
        solver=Pardiso)
    return mesh, survey, sim, m_act, eta_act, act


def main():
    from dctorch import TorchIP3D

    for kind in ("3d-tensor", "3d-tree"):
        mesh, survey, sim, m_dc, eta, act = build_ip_case(kind)
        tag = "actives" if act is not None else "full"
        print(f"[{kind}|{tag}] nC={mesh.nC:,} nD={survey.nD}")

        d_ref = np.asarray(sim.dpred(eta))
        rng = np.random.default_rng(3)
        v = rng.standard_normal(survey.nD)
        g_ref = np.asarray(sim.Jtvec(eta, v))

        for device in ("cuda", "cpu"):
            if device == "cuda" and not torch.cuda.is_available():
                print("[cuda] no disponible, salto")
                continue
            eng = TorchIP3D(mesh, survey, m_dc, device=device,
                            active_cells=act,
                            val_inactive=None if act is None
                            else np.log(1e8))
            et = torch.tensor(eta, device=device)
            d_t = eng.dpred(et).cpu().numpy()
            rel = np.linalg.norm(d_t - d_ref) / np.linalg.norm(d_ref)
            print(f"  [{device}] paridad dpred IP: rel {rel:.2e}")
            assert rel < 1e-10

            et = torch.tensor(eta, device=device, requires_grad=True)
            vt = torch.tensor(v, device=device)
            (eng.dpred(et) * vt).sum().backward()
            g_t = et.grad.cpu().numpy()
            relg = np.linalg.norm(g_t - g_ref) / np.linalg.norm(g_ref)
            print(f"  [{device}] paridad J^T v: rel {relg:.2e}")
            assert relg < 1e-10

            dobs = torch.tensor(d_ref * (1 + 0.05 * rng.standard_normal(
                len(d_ref))), device=device)
            w = torch.tensor(1.0 / (0.05 * np.abs(d_ref) + 1e-6),
                             device=device)
            p = torch.tensor(rng.standard_normal(len(eta)), device=device)
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
            print(f"  [{device}] FD misfit: rel {err:.2e}")
            assert err < 1e-7

    print("test_ip3d OK")


if __name__ == "__main__":
    main()
