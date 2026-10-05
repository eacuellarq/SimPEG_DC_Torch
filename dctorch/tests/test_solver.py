"""
Validación Fase 1: dctorch.solver vs scipy (exactitud) y vs FD (gradientes).
Correr desde la raíz del repo:  python -m dctorch.tests.test_solver
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from dctorch import CuDSSSolver, sparse_solve

DEV = "cuda"


def make_problem(nx=40, n_rhs=5, seed=0):
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(nx, nx))
    I = sp.eye(nx)
    A = (sp.kron(I, T) + sp.kron(T, I) + 0.1 * sp.eye(nx * nx)).tocsr()
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((A.shape[0], n_rhs))
    return A, b


def test_correctness_vs_scipy():
    A, b = make_problem()
    n = A.shape[0]
    slv = CuDSSSolver(torch.tensor(A.indptr, dtype=torch.int32),
                      torch.tensor(A.indices, dtype=torch.int32),
                      n, b.shape[1], device=DEV)
    vals = torch.tensor(A.data, device=DEV)
    bt = torch.tensor(b, device=DEV)
    x = sparse_solve(vals, bt, slv).cpu().numpy()
    x_ref = spla.spsolve(A.tocsc(), b)
    rel = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    print(f"[exactitud] rel-err vs scipy: {rel:.2e}")
    assert rel < 1e-11
    slv.free()


def test_gradients_vs_fd():
    A, b = make_problem(nx=12, n_rhs=3)          # chico: FD denso barato
    n = A.shape[0]
    slv = CuDSSSolver(torch.tensor(A.indptr, dtype=torch.int32),
                      torch.tensor(A.indices, dtype=torch.int32),
                      n, b.shape[1], device=DEV)
    rng = np.random.default_rng(1)
    w = torch.tensor(rng.standard_normal((n, b.shape[1])), device=DEV)

    vals = torch.tensor(A.data, device=DEV, requires_grad=True)
    bt = torch.tensor(b, device=DEV, requires_grad=True)
    loss = (w * sparse_solve(vals, bt, slv)).sum()
    loss.backward()

    def loss_np(data, bb):
        Am = sp.csr_matrix((data, A.indices, A.indptr), shape=A.shape)
        return float((w.cpu().numpy() * spla.spsolve(Am.tocsc(), bb)).sum())

    eps = 1e-6
    idx = rng.choice(A.nnz, 12, replace=False)
    fd, an = [], []
    for p in idx:
        d = A.data.copy(); d[p] += eps
        fd.append((loss_np(d, b) - loss_np(A.data, b)) / eps)
        an.append(float(vals.grad[p]))
    rel_v = np.max(np.abs(np.array(fd) - np.array(an))
                   / (np.abs(fd) + 1e-12))
    ib = [(rng.integers(n), rng.integers(b.shape[1])) for _ in range(6)]
    fdb, anb = [], []
    for i, j in ib:
        bb = b.copy(); bb[i, j] += eps
        fdb.append((loss_np(A.data, bb) - loss_np(A.data, b)) / eps)
        anb.append(float(bt.grad[i, j]))
    rel_b = np.max(np.abs(np.array(fdb) - np.array(anb))
                   / (np.abs(fdb) + 1e-12))
    print(f"[gradientes] dL/dvals vs FD: {rel_v:.2e} | dL/db vs FD: {rel_b:.2e}")
    assert rel_v < 1e-5 and rel_b < 1e-5
    slv.free()


def test_narrow_rhs_and_iteration_speed():
    A, b = make_problem(nx=95, n_rhs=140)        # escala del gate
    n = A.shape[0]
    slv = CuDSSSolver(torch.tensor(A.indptr, dtype=torch.int32),
                      torch.tensor(A.indices, dtype=torch.int32),
                      n, 140, device=DEV)
    vals = torch.tensor(A.data, device=DEV)
    bt = torch.tensor(b, device=DEV)
    x = sparse_solve(vals, bt, slv)
    xn = sparse_solve(vals, bt[:, :7], slv)      # RHS angosto por el mismo plan
    dn = float(torch.linalg.norm(xn - x[:, :7]) / torch.linalg.norm(x[:, :7]))
    print(f"[rhs angosto] coincide con el ancho: {dn:.2e}")
    assert dn < 1e-13

    vals_g = vals.clone().requires_grad_(True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    REP = 20
    for _ in range(REP):
        xx = sparse_solve(vals_g * 1.000001, bt, slv)
        xx.sum().backward()
        vals_g.grad = None
    torch.cuda.synchronize()
    dt = 1e3 * (time.perf_counter() - t0) / REP
    print(f"[iteracion] fwd+bwd (factor+2 solves+grad-gather, n={n:,}, "
          f"140 rhs): {dt:.1f} ms")
    slv.free()


if __name__ == "__main__":
    test_correctness_vs_scipy()
    test_gradients_vs_fd()
    test_narrow_rhs_and_iteration_speed()
    print("FASE 1 VALIDADA")
