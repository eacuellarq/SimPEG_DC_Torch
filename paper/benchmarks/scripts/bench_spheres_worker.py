"""
bench_spheres_worker.py — UN punto del escalado 3D REPRODUCIBLE (problema
dos-esferas del tutorial, 696 datos fijos): mide por fase (mediana de
NCLOSURES warm, cuda.synchronize por borde):
  t_fwd_ng (forward no_grad) | t_fwd_g (forward con grafo) | t_bwd
  (backward) | memorias de inferencia y gradiente | NVML pico (hilo 50ms).
Env: DH (obligatorio), DCTORCH_PREC=f32|f64, NCLOSURES (def 3).
Emite "JSON: {...}". Correr bajo el driver (guard NVML).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from spheres_common import build_problem, starting_logrho
from dctorch import TorchDC3D

DH = float(os.environ["DH"])
PREC = os.environ.get("DCTORCH_PREC", "f32")
DT = {"f64": torch.float64, "f32": torch.float32}[PREC]
NCLO = int(os.environ.get("NCLOSURES", "3"))
torch.manual_seed(0)
np.random.seed(0)

import pynvml
pynvml.nvmlInit()
_h = pynvml.nvmlDeviceGetHandleByIndex(0)
_peak, _stop = [0.0], [False]


def _sample():
    while not _stop[0]:
        _peak[0] = max(_peak[0],
                       pynvml.nvmlDeviceGetMemoryInfo(_h).used / 2**30)
        time.sleep(0.05)


threading.Thread(target=_sample, daemon=True).start()

mesh, survey, dobs, std, active, nC = build_problem(DH)
print(f"[mesh] dh={DH:g} | cells {mesh.n_cells:,} | active {nC:,} | "
      f"nodes {mesh.n_nodes:,} | nD {survey.nD}", flush=True)

t0 = time.time()
eng = TorchDC3D(mesh, survey, device="cuda", dtype=DT,
                active_cells=active, val_inactive=np.log(1e8))
t_build = time.time() - t0
print(f"[build] {t_build:.1f}s", flush=True)

W_t = torch.tensor(1.0 / std, dtype=torch.float64, device="cuda")
dobs_t = torch.tensor(dobs, dtype=torch.float64, device="cuda")
m = torch.tensor(starting_logrho(survey, dobs) * np.ones(nC),
                 dtype=torch.float64, device="cuda", requires_grad=True)


def loss_of(d):
    return 0.5 * torch.sum((W_t * (d.to(torch.float64) - dobs_t)) ** 2)


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    r = fn()
    torch.cuda.synchronize()
    return r, time.time() - t0


t0 = time.time()
loss = loss_of(eng.dpred(m))
loss.backward()
torch.cuda.synchronize()
t_cold = time.time() - t0
m.grad = None

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
fwd_ng = []
for _ in range(NCLO):
    with torch.no_grad():
        _, t = timed(lambda: eng.dpred(m))
    fwd_ng.append(t)
mem_fwd = torch.cuda.max_memory_reserved() / 2**30

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
fwd_g, bwd = [], []
for _ in range(NCLO):
    m.grad = None
    loss, t1 = timed(lambda: loss_of(eng.dpred(m)))
    _, t2 = timed(loss.backward)
    fwd_g.append(t1)
    bwd.append(t2)
mem_grad = torch.cuda.max_memory_reserved() / 2**30
med = lambda x: round(float(np.median(x)), 3)  # noqa: E731
time.sleep(0.2)
_stop[0] = True

print("JSON: " + json.dumps(dict(
    dh=DH, prec=PREC, problem="spheres", n_cells=int(mesh.n_cells),
    n_active=nC, n_nodes=int(mesh.n_nodes), nD=int(survey.nD),
    t_build=round(t_build, 1), t_cold=round(t_cold, 2),
    t_fwd_ng=med(fwd_ng), t_fwd_g=med(fwd_g), t_bwd=med(bwd),
    t_grad_total=round(med(fwd_g) + med(bwd), 3),
    mem_fwd_gib=round(mem_fwd, 2), mem_grad_gib=round(mem_grad, 2),
    nvml_peak_gib=round(_peak[0], 2))), flush=True)
