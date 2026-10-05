"""
Validacion del motor de RESISTIVIDAD COMPLEJA (TorchCR2D), cuda y cpu.

  A. limite DC: con m=0 el forward complejo debe reducirse EXACTAMENTE al DC
     real de stock (|V*| == dpred de Simulation2DNodal).
  B. gate ANALITICO: en semiespacio homogeneo con m uniforme,
     Im(V*)/Re(V*) = m EXACTAMENTE, para CUALQUIER m (no solo m pequeno).
     Ninguna otra formulacion de la casa tiene un gate exacto asi.
  C. consistencia m->0 contra el linealizado de TorchIP2D: error O(m^2)
     -- OJO, un orden MEJOR que la formulacion de dos estados (que da O(m)).
     Razon: expandiendo (A0 - i*dA)^-1, el termino O(m^2) es REAL, o sea que
     cae entero en Re(V*) (la magnitud) y deja el cociente Im/Re correcto un
     orden mas. En dos estados el termino O(eta^2) SI contamina el cociente
     (d = d_lin - d_lin^2 + ...). Medido: razon 4.00 vs 2.02.
  D. gradiente conjunto vs FD en AMBOS bloques -> valida el backward de
     Wirtinger (adj = A^-1 conj(G), grad = -conj(adj*x)).

Correr desde la raiz del repo:  python -m dctorch.tests.test_cr2d
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


def build(data_type):
    from discretize import TensorMesh
    from simpeg.electromagnetics.static import resistivity as dc

    mesh = TensorMesh([[(1.0, 100)], [(1.0, 50)]], origin="CN")
    xe = np.linspace(-30, 30, 21)
    srcs = []
    for i in range(len(xe) - 1):
        lm, ln = [], []
        for n in range(1, 7):
            jm, jn = i + 1 + n, i + 2 + n
            if jn >= len(xe):
                break
            lm.append([xe[jm], 0.0])
            ln.append([xe[jn], 0.0])
        if lm:
            rx = dc.receivers.Dipole(np.array(lm), np.array(ln),
                                     data_type=data_type)
            srcs.append(dc.sources.Dipole([rx], np.r_[xe[i], 0.0],
                                          np.r_[xe[i + 1], 0.0]))
    return mesh, dc.survey.Survey(srcs)


def main():
    from dctorch import TorchCR2D, TorchIP2D

    mesh, survey_v = build("volt")
    _, survey_c = build("apparent_chargeability")
    nC = mesh.nC
    cc = mesh.cell_centers
    m_dc = np.log(100.0) * np.ones(nC)
    blk = (np.abs(cc[:, 0] - 5) < 8) & (cc[:, 1] < -3) & (cc[:, 1] > -10)
    m_dc[blk] = np.log(10.0)
    shape = np.zeros(nC)
    blk2 = (np.abs(cc[:, 0] + 6) < 6) & (cc[:, 1] < -2) & (cc[:, 1] > -8)
    shape[blk2] = 1.0
    print(f"mesh {mesh.shape_cells} nC={nC:,} | nD={survey_v.nD} | nky=11")

    from pymatsolver import Pardiso
    from simpeg import maps
    from simpeg.electromagnetics.static import resistivity as dcmod
    sim = dcmod.Simulation2DNodal(mesh, survey=survey_v,
                                  rhoMap=maps.ExpMap(mesh), nky=11,
                                  solver=Pardiso)
    d_dc_ref = np.asarray(sim.dpred(m_dc))
    m_homo = np.log(100.0) * np.ones(nC)
    d_homo_ref = np.asarray(sim.dpred(m_homo))

    for device in ("cuda", "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            print("[cuda] no disponible, salto")
            continue
        rng = np.random.default_rng(5)
        eng = TorchCR2D(mesh, survey_v, nky=11, device=device, quadrature="simpeg")
        mr = torch.tensor(m_dc, device=device)

        # ---- A. limite DC (m = 0) -----------------------------------------
        d_r, d_i = eng.dpred_from_fields(mr, torch.zeros(nC, device=device))
        rel = np.linalg.norm(np.abs(d_r.cpu().numpy()) - np.abs(d_dc_ref)) \
            / np.linalg.norm(d_dc_ref)
        print(f"[{device}] A. limite DC (m=0) vs stock: rel {rel:.2e} | "
              f"|d_ip| max {float(d_i.abs().max()):.2e}")
        assert rel < 1e-10 and float(d_i.abs().max()) < 1e-12

        # ---- B. gate ANALITICO homogeneo: Im/Re = m exacto -----------------
        print(f"[{device}] B. gate analitico (semiespacio homogeneo)")
        mh = torch.tensor(m_homo, device=device)
        # el error crece como 1/m: Im(V*) es m veces menor que Re(V*) y ambos
        # salen de una DIFERENCIA de potenciales (dipolo receptor) -> la
        # cancelacion se amplifica al dividir. Es una propiedad del dato, no
        # del solver, y la comparte la formulacion de dos estados.
        for mval in (0.001, 0.01, 0.1, 0.4, 0.9, 2.0):
            _, di = eng.dpred_from_fields(
                mh, mval * torch.ones(nC, device=device))
            err = float((di - mval).abs().max() / mval)
            print(f"      m = {mval:6.3f} -> Im/Re error rel max {err:.2e}"
                  f"   (x m = {err*mval:.1e})")
            assert err < 1e-5, "el gate analitico deberia ser ~exacto"

        # ---- C. consistencia m->0 vs linealizado ---------------------------
        lin = TorchIP2D(mesh, survey_c, m_dc, nky=11, device=device, quadrature="simpeg")
        print(f"[{device}] C. gate m->0 (complejo vs linealizado): O(m^2)")
        errs = []
        for amp in (0.08, 0.04, 0.02, 0.01):
            e = torch.tensor(amp * shape, device=device)
            _, d_nl = eng.dpred_from_fields(mr, e)
            d_l = lin.dpred(e)
            r = float(torch.norm(d_nl - d_l) / torch.norm(d_l))
            errs.append(r)
            ratio = errs[-2] / r if len(errs) > 1 else float("nan")
            print(f"      m {amp:5.3f}  rel {r:.3e}  razon {ratio:5.2f}"
                  f"  (esperado ~4 = O(m^2))")
        for k in range(1, len(errs)):
            assert 3.5 < errs[k - 1] / errs[k] < 4.5, "no es O(m^2)"

        # ---- D. gradiente vs FD (valida el backward de Wirtinger) ----------
        m_true = 0.30 * shape + 0.02
        with torch.no_grad():
            o_r, o_i = eng.dpred_from_fields(
                mr, torch.tensor(m_true, device=device))
        o_r, o_i = o_r.cpu().numpy(), o_i.cpu().numpy()
        O_r = torch.tensor(o_r * (1 + 0.02 * rng.standard_normal(o_r.size)),
                           device=device)
        O_i = torch.tensor(o_i + 0.005 * rng.standard_normal(o_i.size),
                           device=device)
        W_r = torch.tensor(1.0 / (0.02 * np.abs(o_r) + 1e-9), device=device)
        W_i = torch.tensor(1.0 / (0.005 * np.ones_like(o_i)), device=device)
        m0 = np.concatenate([m_dc + 0.02 * rng.standard_normal(nC),
                             m_true + 0.01 * rng.standard_normal(nC)])
        m = torch.tensor(m0, device=device, requires_grad=True)
        phi = eng.misfit(m, O_r, W_r, O_i, W_i)
        phi.backward()
        g = m.grad.clone()
        for k in range(3):
            p = np.zeros(2 * nC)
            if k == 0:
                p[:nC] = rng.standard_normal(nC)
            elif k == 1:
                p[nC:] = rng.standard_normal(nC)
            else:
                p = rng.standard_normal(2 * nC)
            p /= np.linalg.norm(p)
            pt = torch.tensor(p, device=device)
            gp = float((g * pt).sum())
            best, row = np.inf, []
            for eps in (1e-2, 1e-3, 1e-4, 1e-5):
                with torch.no_grad():
                    fp = float(eng.misfit(m + eps * pt, O_r, W_r, O_i, W_i))
                    fm = float(eng.misfit(m - eps * pt, O_r, W_r, O_i, W_i))
                err = abs((fp - fm) / (2 * eps) - gp) / abs(gp)
                row.append(f"{eps:g}:{err:.1e}")
                best = min(best, err)
            tag = ("rho", "m  ", "mix")[k]
            print(f"[{device}] D. FD ({tag}): g.p {gp:.6e} phi "
                  f"{float(phi.detach()):.4e} | " + " ".join(row))
            assert best < 1e-6, "el backward complejo (Wirtinger) esta mal"

    print("test_cr2d OK")


if __name__ == "__main__":
    main()
