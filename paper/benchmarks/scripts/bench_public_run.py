"""
bench_public_run.py — driver del RE-BASEO REPRODUCIBLE (PLAN_REPRODUCIBLE):
todo sobre datos publicos de SimPEG, secuencial (timings limpios), guard
NVML en los puntos GPU, resume por jsonl.

  1. escalado por fases dos-esferas: dh 60..12 x {f32,f64}
     -> results/bench_scale3d_spheres.jsonl
  2. vanilla por evaluacion dos-esferas: dh 60..25
     -> results/bench_vanilla3d_spheres.jsonl
  3. GN storeJ RAM dos-esferas: dh 60/35/25 con MAXITER=3
     -> results/gn_storej_ram_spheres.json
  4. ablacion de optimizadores Century -> results/ablation_century.jsonl

Correr (CWD _gpubench, env simpeg311): python -u bench_public_run.py
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pynvml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), r"paper\benchmarks"))
RES = os.path.join(REPO, "results")
LIMIT = float(os.environ.get("GUARD_GB", "6.5"))
pynvml.nvmlInit()
H = pynvml.nvmlDeviceGetHandleByIndex(0)


def run_child(script, env_extra, guard=True):
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", **env_extra)
    child = subprocess.Popen([sys.executable, "-u",
                              os.path.join(HERE, script)], env=env,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
    res = [None]

    def guard_fn():
        while child.poll() is None:
            used = pynvml.nvmlDeviceGetMemoryInfo(H).used / 2**30
            if used > LIMIT:
                print(f"[GUARD] {used:.2f} GiB > {LIMIT} — KILL", flush=True)
                child.kill()
                return
            time.sleep(1.0)

    if guard:
        threading.Thread(target=guard_fn, daemon=True).start()
    for line in child.stdout:
        line = line.rstrip()
        if line.startswith(("JSON:", "[mesh]", "[build]", "[GUARD]")):
            print("  " + line, flush=True)
        if line.startswith("JSON: "):
            res[0] = json.loads(line[6:])
    child.wait()
    return res[0], child.returncode


def sweep_jsonl(path, points, script, env_of):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {tuple(json.loads(ln)[k] for k in ("dh", "prec"))
                    if "prec" in json.loads(ln) else (json.loads(ln)["dh"],)
                    for ln in open(path)}
    for p in points:
        key = p if isinstance(p, tuple) else (p,)
        if key in done:
            continue
        print(f"\n=== {script} {p} ===", flush=True)
        res, code = run_child(script, env_of(p))
        if res is None:
            res = dict(dh=key[0], killed=True, exit=code,
                       **({"prec": key[1]} if len(key) > 1 else {}))
            print("  [punto MUERTO]", flush=True)
        with open(path, "a") as f:
            f.write(json.dumps(res) + "\n")


# 1) escalado por fases (GPU)
sweep_jsonl(os.path.join(RES, "bench_scale3d_spheres.jsonl"),
            [(dh, prec)
             for dh in (60.0, 45.0, 35.0, 30.0, 25.0, 20.0, 17.0, 15.0,
                        13.0, 12.0)
             for prec in ("f32", "f64")],
            "bench_spheres_worker.py",
            lambda p: dict(DH=f"{p[0]:g}", DCTORCH_PREC=p[1]))

# 2) vanilla por evaluacion (CPU)
sweep_jsonl(os.path.join(RES, "bench_vanilla3d_spheres.jsonl"),
            [90.0, 60.0, 45.0, 35.0, 30.0, 25.0],
            "bench_spheres_vanilla_worker.py",
            lambda p: dict(DH=f"{p:g}"))

# 3) GN storeJ RAM (CPU, 3 iters — el pico ocurre al formar J en la 1a)
gn_path = os.path.join(RES, "gn_storej_ram_spheres.json")
if not os.path.exists(gn_path):
    rows = []
    for dh in (60.0, 35.0, 25.0):
        print(f"\n=== GN RAM dh={dh:g} ===", flush=True)
        res, _ = run_child("inv_tutorial3d_vanilla.py",
                           dict(DH=f"{dh:g}", MAXITER="3"), guard=False)
        rows.append(dict(dh=dh, n_active=res["nC"],
                         ram_gib=res["ram_gib"],
                         note="RSS pico, tutorial GN storeJ, 3 iters"))
    json.dump(rows, open(gn_path, "w"), indent=1)
    print(f"saved {gn_path}", flush=True)

# 4) ablacion Century (GPU)
if not os.path.exists(os.path.join(RES, "ablation_century.jsonl")):
    print("\n=== ablacion Century ===", flush=True)
    run_child("inv_century_ablation.py", {})

print("\nPUBLIC RE-BASE DONE", flush=True)
