"""Driver vanilla: barre DH, un proceso por punto, append a jsonl."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "bench_vanilla3d_results.jsonl")
done = set()
if os.path.exists(OUT):
    done = {json.loads(ln)["dh"] for ln in open(OUT)}
for dh in (90.0, 80.0, 60.0, 50.0, 45.0, 40.0):
    if dh in done:
        continue
    print(f"\n=== vanilla dh={dh:g} ===", flush=True)
    env = dict(os.environ, DH=f"{dh:g}", KMP_DUPLICATE_LIB_OK="TRUE")
    p = subprocess.Popen(
        [sys.executable, "-u",
         os.path.join(HERE, "bench_vanilla3d_worker.py")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")
    res = None
    for line in p.stdout:
        line = line.rstrip()
        if line:
            print("  " + line, flush=True)
        if line.startswith("JSON: "):
            res = json.loads(line[6:])
    p.wait()
    if res is None:
        res = dict(dh=dh, engine="vanilla-pardiso", killed=True,
                   exit=p.returncode)
        print("  [punto MUERTO]", flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(res) + "\n")
print("DONE", flush=True)
