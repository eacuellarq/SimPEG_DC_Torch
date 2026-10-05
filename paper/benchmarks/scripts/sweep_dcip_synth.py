"""
sweep_dcip_synth.py — ablacion del BALANCE del conjunto (el precio conocido
del metodo: 2 misfits + 2 regularizaciones). Corre inv_dcip_synth.py con
distintas combinaciones y resume las lineas JSON.

    python -u sweep_dcip_synth.py           (la bateria completa)
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

B = dict(ETA0="0.05", ETA_MAX="0.6", ALPHA_ETA="1", WARM="0", REG_SPACE="eta")
RUNS = [
    ("base",     dict(B)),
    ("aeta0.1",  dict(B, ALPHA_ETA="0.1")),
    ("aeta10",   dict(B, ALPHA_ETA="10")),
    ("logit",    dict(B, REG_SPACE="logit")),
    ("nobound",  dict(B, ETA0="0.01", ETA_MAX="1.0", REG_SPACE="logit")),
    ("warm",     dict(B, WARM="1")),
]


def main():
    rows = []
    for tag, env in RUNS:
        print(f"\n########## {tag}: {env}", flush=True)
        e = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", TAG="_" + tag, **env)
        p = subprocess.run([sys.executable, "-u", "inv_dcip_synth.py"],
                           cwd=HERE, env=e, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-3000:], p.stderr[-3000:])
            raise SystemExit(f"fallo {tag}")
        line = [l for l in p.stdout.splitlines() if l.startswith("JSON: ")][-1]
        r = json.loads(line[6:])
        r["tag"] = tag
        rows.append(r)
        print("  ".join(l for l in p.stdout.splitlines()
                        if l.startswith("[secuencial") or
                        l.startswith("[conjunto")), flush=True)

    print("\n===== RESUMEN (todos contra la misma verdad, eta=0.40) =====")
    hdr = (f"{'run':10s} {'eta_core':>9s} {'eta_max':>8s} {'eta_corr':>9s} "
           f"{'eta_rms':>8s} {'rho_corr':>9s} {'chi2_dc':>8s} {'chi2_ip':>8s} "
           f"{'outers':>7s} {'s':>6s}")
    print(hdr)
    s = rows[0]["seq"]
    print(f"{'SECUENCIAL':10s} {s['eta_core_mean']:9.4f} {s['eta_max']:8.4f} "
          f"{s['eta_corr']:9.4f} {s['eta_rms']:8.4f} "
          f"{s['rho_corr_total']:9.4f} {s['chi2_dc_total']:8.3f} "
          f"{s['chi2_ip_total']:8.3f} {s['outers']:7d} {s['t']:6.1f}")
    for r in rows:
        j = r["joint"]
        print(f"{r['tag']:10s} {j['eta_core_mean']:9.4f} {j['eta_max']:8.4f} "
              f"{j['eta_corr']:9.4f} {j['eta_rms']:8.4f} {j['rho_corr']:9.4f} "
              f"{j['chi2_dc']:8.3f} {j['chi2_ip']:8.3f} {j['outers']:7d} "
              f"{j['t']:6.1f}")
    with open(os.path.join(os.path.dirname(HERE), "results",
                           "sweep_dcip_synth.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
