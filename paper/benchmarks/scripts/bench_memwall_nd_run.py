"""Driver pared-de-memoria vs N_d (malla fija esferas dh=25, survey denso).
GN hasta 33 lineas (~9k datos); dctorch ademas 65 (~18k). Resume por jsonl."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), r"paper\benchmarks\results\memwall_vs_nd.jsonl"))
POINTS = [(n, e) for n in (3, 9, 17, 33) for e in ("dctorch", "gn")] + \
         [(65, "dctorch")]
done = set()
if os.path.exists(OUT):
    done = {(json.loads(l)["nlines"], json.loads(l)["engine"].split("-")[0])
            for l in open(OUT)}
for nlines, eng in POINTS:
    key = (nlines, "gn" if eng == "gn" else "dctorch")
    if key in done:
        continue
    print(f"=== nlines={nlines} {eng} ===", flush=True)
    env = dict(os.environ, NLINES=str(nlines), ENGINE=eng,
               KMP_DUPLICATE_LIB_OK="TRUE")
    p = subprocess.Popen([sys.executable, "-u",
                          os.path.join(HERE, "bench_memwall_nd_worker.py")],
                         env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    res = None
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith(("JSON:", "[survey]")):
            print("  " + line, flush=True)
        if line.startswith("JSON: "):
            res = json.loads(line[6:])
    p.wait()
    if res:
        with open(OUT, "a") as f:
            f.write(json.dumps(res) + "\n")
print("MEMWALL-ND DONE", flush=True)
