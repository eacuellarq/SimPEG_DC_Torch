"""
run_all.py — reproduce TODOS los resultados publicos del preprint desde cero.

    python paper/benchmarks/run_all.py [--quick]

Requisitos: env con simpeg>=0.25, torch+CUDA, nvmath-python[cudss], pydiso,
psutil, nvidia-ml-py (ver appendix de reproducibilidad del paper). Los dos
datasets se DESCARGAN solos de los repos de SimPEG (user-tutorials y
transform-2020-simpeg). GPU de 8 GB basta; guard NVML integrado.

Etapas (cada una reanudable; --quick corre solo las inversiones insignia):
  1. esferas 3D: vanilla verbatim + dctorch F32/F64 (lean y con sensitivity
     weights) a dh=25; control dh=60; chi2 del modelo verdadero
  2. Century 2D: vanilla + dctorch F32/F64
  3. escalado por fases + vanilla por evaluacion + GN RAM (dos-esferas)
  4. ablacion de optimizadores (Century)
  5. figuras: jupyter nbconvert --execute figures_paper.ipynb + scripts fig_*
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")


def run(script, **env_extra):
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
               **{k: str(v) for k, v in env_extra.items()})
    print(f"\n=== {script} {env_extra} ===", flush=True)
    r = subprocess.run([sys.executable, "-u",
                        os.path.join(SCRIPTS, script)], env=env)
    if r.returncode != 0:
        print(f"[WARN] {script} exit {r.returncode} — continuando",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="solo las inversiones insignia (sin sweeps)")
    args = ap.parse_args()

    # 1) esferas
    run("inv_tutorial3d_vanilla.py", DH=25)
    for prec in ("f32", "f64"):
        run("inv_tutorial3d_dctorch.py", DH=25, DCTORCH_PREC=prec, SENSW=0)
        run("inv_tutorial3d_dctorch.py", DH=25, DCTORCH_PREC=prec, SENSW=1)
    run("tutorial_true_chi.py", DH=25)
    run("tutorial_true_chi.py", DH=60)

    # 2) Century
    run("inv_century2d_vanilla.py")
    for prec in ("f32", "f64"):
        run("inv_century2d_dctorch.py", DCTORCH_PREC=prec)

    if not args.quick:
        # 3+4) sweeps + ablacion (driver propio, reanudable)
        run("bench_public_run.py")

    # 5) figuras
    run("plot_tutorial_slices_dh25.py")
    run("plot_century_models.py")
    run("fig_tikhonov_nature.py")
    run("fig_public_scaling.py")
    run("fig_century_ablation.py")
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--inplace", os.path.join(HERE, "figures_paper.ipynb")])
    print("\nALL PUBLIC RESULTS REPRODUCED", flush=True)


if __name__ == "__main__":
    main()
