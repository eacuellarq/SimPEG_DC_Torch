"""
Validacion del IP CONJUNTO NO-LINEAL (TorchDCIP2D), en cuda y cpu.

SimPEG stock NO tiene una simulacion IP no-lineal, asi que no hay paridad
directa contra que medir. Los tres gates son:

  A. los DOS estados del batch vs stock (paridad maquina) con eta != 0:
     A1 dc_state="instantaneous" -> d_DC == F(sigma) (eta no se cuela al DC)
     A2 dc_state="total" -> d_DC == F(sigma(1-eta)): el segundo sistema del
     batch ES el DC del medio polarizado.
  B. consistencia eta->0: d_IP no-lineal vs el linealizado de TorchIP2D
     (mismo fondo) tiene error relativo O(eta) — al dividir eta entre 2 el
     error se divide entre ~2 (orden de convergencia medido, no asumido).
  C. gradiente conjunto vs diferencias finitas centradas en AMBOS bloques
     (log-rho y logit-eta) con direcciones aleatorias.

Correr desde la raiz del repo:  python -m dctorch.tests.test_dcip2d
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch


def build_problem(data_type):
    """Malla/survey del test_ip2d (dipolo-dipolo n=1..6) con data_type dado."""
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


def models(mesh):
    m_dc = np.log(100.0) * np.ones(mesh.nC)
    cc = mesh.cell_centers
    blk = (np.abs(cc[:, 0] - 5) < 8) & (cc[:, 1] < -3) & (cc[:, 1] > -10)
    m_dc[blk] = np.log(10.0)
    eta = np.zeros(mesh.nC)
    blk2 = (np.abs(cc[:, 0] + 6) < 6) & (cc[:, 1] < -2) & (cc[:, 1] > -8)
    eta[blk2] = 1.0                      # amplitud la fija el gate
    return m_dc, eta


def main():
    from dctorch import TorchDCIP2D, TorchIP2D

    mesh, survey_v = build_problem("volt")            # el conjunto
    _, survey_c = build_problem("apparent_chargeability")   # el linealizado
    m_dc, shape_eta = models(mesh)
    nC = mesh.nC
    print(f"mesh {mesh.shape_cells} nC={nC:,} | nD={survey_v.nD} | nky=11")

    # referencia DC stock
    from pymatsolver import Pardiso
    from simpeg import maps
    from simpeg.electromagnetics.static import resistivity as dcmod
    sim_dc = dcmod.Simulation2DNodal(mesh, survey=survey_v,
                                     rhoMap=maps.ExpMap(mesh), nky=11,
                                     solver=Pardiso)
    eta_A = 0.35 * shape_eta + 0.02              # eta GRANDE para el gate A
    d_inf_ref = np.asarray(sim_dc.dpred(m_dc))                  # F(sigma)
    m_tot = -np.log(np.exp(-m_dc) * (1.0 - eta_A))
    d_tot_ref = np.asarray(sim_dc.dpred(m_tot))            # F(sigma(1-eta))

    for device in ("cuda", "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            print("[cuda] no disponible, salto")
            continue
        rng = np.random.default_rng(11)       # mismas direcciones en ambos
        eng = TorchDCIP2D(mesh, survey_v, nky=11, device=device, quadrature="simpeg")
        lin = TorchIP2D(mesh, survey_c, m_dc, nky=11, device=device, quadrature="simpeg")
        mr = torch.tensor(m_dc, device=device)

        # ---- A. los dos estados del batch vs stock --------------------------
        etA = torch.tensor(eta_A, device=device)
        for state, ref in (("instantaneous", d_inf_ref), ("total", d_tot_ref)):
            eng.dc_state = state
            d_dc, _ = eng.dpred_from_fields(mr, etA)
            rel = np.linalg.norm(d_dc.cpu().numpy() - ref) \
                / np.linalg.norm(ref)
            print(f"[{device}] A. estado {state:13s} vs stock: rel {rel:.2e}")
            assert rel < 1e-10
        eng.dc_state = "total"

        # ---- B. consistencia eta->0 vs linealizado -------------------------
        print(f"[{device}] B. gate eta->0 (no-lineal vs linealizado)")
        errs = []
        for amp in (0.08, 0.04, 0.02, 0.01):
            e = torch.tensor(amp * shape_eta, device=device)
            _, d_nl = eng.dpred_from_fields(mr, e)
            d_l = lin.dpred(e)
            r = float(torch.norm(d_nl - d_l) / torch.norm(d_l))
            errs.append(r)
            ratio = errs[-2] / r if len(errs) > 1 else float("nan")
            print(f"      eta_max {amp:5.3f}  rel {r:.3e}  "
                  f"razon vs anterior {ratio:5.2f}  (esperado ~2)")
        for k in range(1, len(errs)):
            assert 1.7 < errs[k - 1] / errs[k] < 2.3, "no es O(eta)"
        assert errs[-1] < 1e-2

        # ---- C. gradiente conjunto vs FD ------------------------------------
        eta_true = 0.30 * shape_eta + 0.02
        et = torch.tensor(eta_true, device=device)
        with torch.no_grad():
            o_dc, o_ip = eng.dpred_from_fields(mr, et)
        o_dc = o_dc.cpu().numpy()
        o_ip = o_ip.cpu().numpy()
        dobs_dc = torch.tensor(o_dc * (1 + 0.02 * rng.standard_normal(o_dc.size)),
                               device=device)
        dobs_ip = torch.tensor(o_ip * (1 + 0.05 * rng.standard_normal(o_ip.size)),
                               device=device)
        w_dc = torch.tensor(1.0 / (0.02 * np.abs(o_dc) + 1e-8), device=device)
        w_ip = torch.tensor(1.0 / (0.05 * np.abs(o_ip) + 1e-6), device=device)

        # arranque cerca del verdadero: phi_d moderado, si no el piso de la FD
        # lo fija el ruido de reduccion de la GPU (~1e-13 * phi)
        m0 = np.concatenate([m_dc + 0.02 * rng.standard_normal(nC),
                             eng.m_eta_of(eta_true)
                             + 0.05 * rng.standard_normal(nC)])
        m = torch.tensor(m0, device=device, requires_grad=True)
        phi = eng.misfit(m, dobs_dc, w_dc, dobs_ip, w_ip)
        phi.backward()
        g = m.grad.clone()

        for k in range(3):
            p = np.zeros(2 * nC)
            if k == 0:                       # solo bloque rho
                p[:nC] = rng.standard_normal(nC)
            elif k == 1:                     # solo bloque eta
                p[nC:] = rng.standard_normal(nC)
            else:                            # mezclado
                p = rng.standard_normal(2 * nC)
            p /= np.linalg.norm(p)           # direccion unitaria: el error de
            pt = torch.tensor(p, device=device)   # truncacion baja como eps^2
            gp = float((g * pt).sum())
            tag = ("rho", "eta", "mixto")[k]
            best = np.inf
            row = []
            for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
                with torch.no_grad():
                    fp = float(eng.misfit(m + eps * pt, dobs_dc, w_dc,
                                          dobs_ip, w_ip))
                    fm = float(eng.misfit(m - eps * pt, dobs_dc, w_dc,
                                          dobs_ip, w_ip))
                err = abs((fp - fm) / (2 * eps) - gp) / abs(gp)
                row.append(f"{eps:g}:{err:.1e}")
                best = min(best, err)
            print(f"[{device}] C. FD ({tag:5s}): g.p {gp:.6e} phi "
                  f"{float(phi.detach()):.4e} | " + " ".join(row))
            # piso de la FD: el ruido de redondeo del solve (~1e-13*phi en
            # cuDSS, algo mas en SuperLU) dividido por eps*g.p — por eso se
            # barre eps y se toma el minimo
            assert best < 1e-6

    print("test_dcip2d OK")


if __name__ == "__main__":
    main()
