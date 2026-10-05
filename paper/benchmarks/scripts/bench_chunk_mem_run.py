"""Drive bench_chunk_mem_worker.py: one cold process per point.

Sweeps the chunk width for the forward alone and for the forward-plus-gradient,
so the memory can be attributed: what the fields cost, what differentiability
adds on top, and how much of each chunking removes.

Run, CWD = this directory:
  cmd /c "call activate simpeg311 && python -u bench_chunk_mem_run.py"
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "chunk_memory.jsonl")
NLINES = os.environ.get("NLINES", "65")
CHUNKS = [int(c) for c in os.environ.get("CHUNKS", "0,1024,512,256,128,64").split(",")]
MODES = os.environ.get("MODES", "fwd,grad").split(",")

rows = []
for mode in MODES:
    for c in CHUNKS:
        env = dict(os.environ, NLINES=NLINES, CHUNK=str(c), MODE=mode)
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        p = subprocess.run([sys.executable, "-u", "bench_chunk_mem_worker.py"],
                           cwd=HERE, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        line = next((l for l in p.stdout.splitlines()
                     if l.startswith("JSON: ")), None)
        if line is None:
            print(f"[fail] mode={mode} chunk={c} rc={p.returncode}")
            print((p.stdout + p.stderr)[-600:])
            continue
        r = json.loads(line[6:])
        rows.append(r)
        print(f"  {mode:4s} chunk {r['chunk']:>5} | {r['t_s']:>6.3f} s | "
              f"peak {r['nvml_peak_gib']:>6.3f} | base {r['nvml_base_gib']:>6.3f}"
              f" | delta {r['nvml_delta_gib']:>6.3f} GiB", flush=True)

with open(OUT, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

print("\n" + "=" * 72)
print(f"{'chunk':>7} {'fwd peak':>10} {'grad peak':>10} {'AD extra':>10} "
      f"{'fwd s':>8} {'grad s':>8}")
byc = {}
for r in rows:
    byc.setdefault(r["chunk"], {})[r["mode"]] = r
for c in sorted(byc, reverse=True):
    f, g = byc[c].get("fwd"), byc[c].get("grad")
    if not (f and g):
        continue
    print(f"{c:>7} {f['nvml_peak_gib']:>10.3f} {g['nvml_peak_gib']:>10.3f} "
          f"{g['nvml_peak_gib'] - f['nvml_peak_gib']:>10.3f} "
          f"{f['t_s']:>8.3f} {g['t_s']:>8.3f}")
print(f"\n-> {OUT}")
