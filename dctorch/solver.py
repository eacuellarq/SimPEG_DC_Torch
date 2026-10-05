"""
Sparse SPD direct solve on GPU (cuDSS via nvmath) as a torch autograd op.

Design (ported from the proven ctypes engine, now on the maintained API):
- One `CuDSSSolver` per sparsity pattern: plan (reordering + symbolic
  factorization) runs ONCE; per iteration only the matrix VALUES change,
  copied in-place into the buffers the DirectSolver references, followed by a
  numeric refactorization. This is the "98% cache" of the old engine as
  first-class API.
- backward reuses the same factorization (A is SPD => adjoint solve is a plain
  solve) and assembles grad(values) only on the sparsity pattern, gathered in
  chunks to bound peak memory (the old VRAM fix, kept).
- cuDSS wants the dense RHS in column-major layout: persistent (nrhs, n)
  buffers exposed as .T views.

Only torch + nvmath here — no simpeg, no scipy at runtime.
"""
import os
import threading

import numpy as np
import torch
# nvmath is the cuDSS binding and is therefore CUDA-only. It is imported
# optionally so that `import dctorch` works on a machine without a GPU: the
# scipy/SuperLU path at the bottom of this module needs none of it, and the
# factories below already dispatch on the device. Asking for a cuda solver
# without nvmath raises, with the reason, at construction.
try:
    from nvmath.sparse.advanced import (
        DirectSolver,
        DirectSolverMatrixType,
        DirectSolverOptions,
    )
except ImportError as _exc:  # pragma: no cover - exercised by CPU-only installs
    DirectSolver = DirectSolverMatrixType = DirectSolverOptions = None
    _NVMATH_ERR = _exc
else:
    _NVMATH_ERR = None

# cuDSS matrix types we expose. SPD is the DC default; a complex-conductivity
# operator is complex SYMMETRIC (A^T = A but NOT Hermitian) — measured
# 2026-07-27: SYMMETRIC gives residual 1.9e-16 on such a system while
# HERMITIAN returns a WRONG answer WITHOUT raising (residual 5e-2).
_MTYPE = {} if DirectSolverMatrixType is None else {
    "spd": DirectSolverMatrixType.SPD,
    "symmetric": DirectSolverMatrixType.SYMMETRIC,
    "general": DirectSolverMatrixType.GENERAL}


def _require_nvmath():
    if _NVMATH_ERR is not None:
        raise ImportError(
            "a cuda solver was requested but nvmath-python (the cuDSS "
            "binding) is not importable, so only the cpu path is available: "
            f"{_NVMATH_ERR}") from _NVMATH_ERR

_tls = threading.local()


def _ensure_cuda_ctx(index=0):
    """cuda.core (nvmath's backend) needs a per-thread current device; torch's
    autograd backward runs on its own thread, so set it lazily per thread."""
    if getattr(_tls, "ready", False):
        return
    try:
        from cuda.core import Device
    except ImportError:                      # older cuda-core layouts
        from cuda.core.experimental import Device
    Device(index).set_current()
    _tls.ready = True

# elements per gather chunk in backward (~64 MB of f64 intermediates)
_GRAD_CHUNK_ELEMS = int(os.environ.get("DCTORCH_GRAD_CHUNK", 8_000_000))


class CuDSSSolver:
    """Stateful SPD solver bound to one sparsity pattern and RHS width.

    Parameters
    ----------
    crow, col : int32/int64 tensors on the target device (CSR pattern, FULL
        matrix — nvmath handles the SPD view internally).
    n_rhs : number of right-hand sides per solve (fixed per instance).
    device, dtype : placement and precision of the factorization/solve
        (cuDSS supports float32/float64).
    """

    def __init__(self, crow, col, n, n_rhs, device="cuda",
                 dtype=torch.float64, matrix_type="spd"):
        _require_nvmath()
        self.n, self.n_rhs = n, n_rhs
        self.device, self.dtype = torch.device(device), dtype
        self._sdt = dtype
        self.crow = crow.to(self.device)
        self.col = col.to(self.device)
        nnz = self.col.numel()
        # persistent buffers the DirectSolver references (values updated
        # in-place; a NEW tensor would invalidate the plan). copy_ casts.
        self._vals = torch.zeros(nnz, dtype=self._sdt, device=self.device)
        self._b = torch.zeros(n_rhs, n, dtype=self._sdt, device=self.device).T
        self._A = torch.sparse_csr_tensor(self.crow, self.col, self._vals,
                                          size=(n, n), device=self.device)
        opts = DirectSolverOptions(sparse_system_type=_MTYPE[matrix_type])
        self._slv = DirectSolver(self._A, self._b, options=opts)
        # measured 2026-07-13 (bench_solver_hparams): superpanels OFF gives
        # -12% factor nnz in 3D and a faster small-mesh closure, same residual
        if os.environ.get("DCTORCH_SUPERPANELS", "1") == "0":
            self._slv.plan_config.use_superpanels = False
        self._slv.plan()
        self._planned_vals = None
        # expanded row index per nnz (for the backward gather)
        counts = self.crow[1:] - self.crow[:-1]
        self.row = torch.repeat_interleave(
            torch.arange(n, device=self.device), counts)

    def factorize(self, values):
        """Numeric refactorization with new values (same pattern)."""
        _ensure_cuda_ctx(self.device.index or 0)
        self._vals.copy_(values)
        self._slv.factorize()

    def solve(self, b):
        """Solve A x = b for the CURRENT factorization. b: (n,) or (n, k<=n_rhs)."""
        _ensure_cuda_ctx(self.device.index or 0)
        b2 = b.reshape(self.n, -1)
        k = b2.shape[1]
        if k == self.n_rhs:
            self._b.copy_(b2)
            x = self._slv.solve()
            return x.to(dtype=self.dtype, copy=True).reshape(b.shape)
        # narrower RHS: pad into the fixed-width buffer
        self._b[:, :k] = b2
        self._b[:, k:] = 0.0
        x = self._slv.solve()
        return x[:, :k].to(dtype=self.dtype, copy=True).reshape(b.shape)

    def free(self):
        self._slv.free()


class _SparseSolveFn(torch.autograd.Function):
    """x = A(values)^{-1} b, differentiable in values and b (A SPD)."""

    @staticmethod
    def forward(ctx, values, b, solver):
        solver.factorize(values.detach())
        x = solver.solve(b.detach())
        ctx.solver = solver
        ctx.dt_vals, ctx.dt_b = values.dtype, b.dtype
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        (x,) = ctx.saved_tensors
        solver = ctx.solver
        # adjoint solve reuses the factorization (SPD => A^T = A)
        adj = solver.solve(grad_x.contiguous())
        grad_vals = None
        if ctx.needs_input_grad[0]:
            xm = x.reshape(solver.n, -1)
            am = adj.reshape(solver.n, -1)
            k = xm.shape[1]
            nnz = solver.col.numel()
            grad_vals = torch.empty(nnz, dtype=xm.dtype, device=xm.device)
            step = max(_GRAD_CHUNK_ELEMS // max(k, 1), 1)
            for p0 in range(0, nnz, step):
                p1 = min(p0 + step, nnz)
                grad_vals[p0:p1] = -(am[solver.row[p0:p1]]
                                     * xm[solver.col[p0:p1]]).sum(-1)
            grad_vals = grad_vals.to(ctx.dt_vals)
        grad_b = adj.to(ctx.dt_b) if ctx.needs_input_grad[1] else None
        return grad_vals, grad_b, None


def sparse_solve(values, b, solver):
    """Differentiable sparse SPD solve. `solver` is a CuDSSSolver for the pattern."""
    return _SparseSolveFn.apply(values, b, solver)


class CuDSSBatchSolver:
    """Uniform-batch SPD solver: `nbatch` systems SHARING one sparsity pattern
    (e.g. the nky systems of 2.5D DC, or model particles), solved as a cuDSS
    SEQUENCE batch — no giant block-diagonal super-pattern, so peak memory is
    per-system. Plan runs once; per iteration only values change (in place)."""

    def __init__(self, crow, col, n, n_rhs, nbatch, device="cuda",
                 dtype=torch.float64, matrix_type="spd"):
        _require_nvmath()
        self.n, self.n_rhs, self.nbatch = n, n_rhs, nbatch
        self.device, self.dtype = torch.device(device), dtype
        self._sdt = dtype
        self.crow = crow.to(self.device)
        self.col = col.to(self.device)
        nnz = self.col.numel()
        self._vals = [torch.zeros(nnz, dtype=self._sdt, device=self.device)
                      for _ in range(nbatch)]
        self._bs = [torch.zeros(n_rhs, n, dtype=self._sdt,
                                device=self.device).T
                    for _ in range(nbatch)]
        As = [torch.sparse_csr_tensor(self.crow, self.col, v, size=(n, n),
                                      device=self.device) for v in self._vals]
        opts = DirectSolverOptions(sparse_system_type=_MTYPE[matrix_type])
        self._slv = DirectSolver(As, self._bs, options=opts)
        if os.environ.get("DCTORCH_SUPERPANELS", "1") == "0":
            self._slv.plan_config.use_superpanels = False
        self._slv.plan()
        counts = self.crow[1:] - self.crow[:-1]
        self.row = torch.repeat_interleave(
            torch.arange(n, device=self.device), counts)
        # bumped by every factorize, so a holder of a factorization (a
        # linearization) can tell that someone else has since replaced it
        self.stamp = 0

    def factorize(self, values):
        """values: (nbatch, nnz) — refactorize every system, same pattern."""
        _ensure_cuda_ctx(self.device.index or 0)
        for i in range(self.nbatch):
            self._vals[i].copy_(values[i])
        self._slv.factorize()
        self.stamp += 1

    def solve(self, b):
        """Solve all systems. b: (nbatch, n, n_rhs) or (n, n_rhs) shared."""
        _ensure_cuda_ctx(self.device.index or 0)
        shared = b.dim() == 2
        for i in range(self.nbatch):
            self._bs[i].copy_(b if shared else b[i])
        xs = self._slv.solve()
        return torch.stack([x.to(dtype=self.dtype, copy=True) for x in xs])

    def free(self):
        self._slv.free()


class _SparseSolveBatchFn(torch.autograd.Function):
    """x_i = A_i(values_i)^{-1} b, all A_i SPD on one shared pattern.

    COMPLEX-SAFE. For a real loss L, torch hands `backward` the conjugate
    Wirtinger derivative G = dL/dx*. With x = A^{-1} b holomorphic in A
    (dx/dA* = 0):

        dL/dA*_ij = -conj(adj_i * x_j),   adj = A^{-1} conj(G)
        dL/db*_i  =  conj(adj_i)

    using A^T = A (symmetric, real SPD or complex symmetric) so the adjoint
    reuses the SAME factorization. `.conj()` is an exact no-op on real
    tensors, so this is one code path for both worlds.
    """

    @staticmethod
    def forward(ctx, values, b, solver):
        solver.factorize(values.detach())
        x = solver.solve(b.detach())
        ctx.solver = solver
        ctx.b_shared = b.dim() == 2
        ctx.dt_vals, ctx.dt_b = values.dtype, b.dtype
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        (x,) = ctx.saved_tensors
        solver = ctx.solver
        adj = solver.solve(grad_x.conj().resolve_conj().contiguous())
        grad_vals = None
        if ctx.needs_input_grad[0]:
            nb, k = solver.nbatch, x.shape[-1]
            nnz = solver.col.numel()
            grad_vals = torch.empty(nb, nnz, dtype=x.dtype, device=x.device)
            step = max(_GRAD_CHUNK_ELEMS // max(nb * k, 1), 1)
            for p0 in range(0, nnz, step):
                p1 = min(p0 + step, nnz)
                grad_vals[:, p0:p1] = -(adj[:, solver.row[p0:p1]]
                                        * x[:, solver.col[p0:p1]]).sum(-1)
            grad_vals = grad_vals.conj().resolve_conj().to(ctx.dt_vals)
        grad_b = None
        if ctx.needs_input_grad[1]:
            gb = adj.sum(0) if ctx.b_shared else adj
            grad_b = gb.conj().resolve_conj().to(ctx.dt_b)
        return grad_vals, grad_b, None


class _SparseSolveBatchPreFn(torch.autograd.Function):
    """Same solve and same gradients as `_SparseSolveBatchFn`, for an A whose
    numeric factorization the CALLER already computed. That lets one
    factorization serve several right-hand-side blocks, which is what makes it
    affordable to split a survey into source chunks."""

    @staticmethod
    def forward(ctx, values, b, solver):
        x = solver.solve(b.detach())
        ctx.solver = solver
        ctx.b_shared = b.dim() == 2
        ctx.dt_vals, ctx.dt_b = values.dtype, b.dtype
        ctx.save_for_backward(x)
        return x

    backward = _SparseSolveBatchFn.backward


def sparse_solve_batch_prefactored(values, b, solver):
    """Differentiable solve on an already-factorized A (see the class above)."""
    return _SparseSolveBatchPreFn.apply(values, b, solver)


def sparse_solve_batch(values, b, solver):
    """Differentiable batched SPD solve on a shared pattern.

    values: (nbatch, nnz) | b: (nbatch, n, n_rhs) or (n, n_rhs) shared."""
    return _SparseSolveBatchFn.apply(values, b, solver)


class SciPySolver:
    """CPU drop-in for CuDSSSolver: scipy SuperLU behind the same interface,
    so the autograd Functions (and the simulations) are device-agnostic."""

    def __init__(self, crow, col, n, n_rhs, device="cpu",
                 dtype=torch.float64):
        assert torch.device(device).type == "cpu"
        self.n, self.n_rhs = n, n_rhs
        self.device, self.dtype = torch.device("cpu"), dtype
        self.crow = crow.to("cpu")
        self.col = col.to("cpu")
        self._indptr = self.crow.numpy()
        self._indices = self.col.numpy()
        counts = self.crow[1:] - self.crow[:-1]
        self.row = torch.repeat_interleave(torch.arange(n), counts)
        self._lu = None

    def factorize(self, values):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        A = sp.csr_matrix((values.detach().cpu().numpy(), self._indices,
                           self._indptr), shape=(self.n, self.n))
        self._lu = spla.splu(A.tocsc())

    def solve(self, b):
        x = self._lu.solve(np.ascontiguousarray(
            b.detach().cpu().numpy().reshape(self.n, -1)))
        return torch.as_tensor(x, dtype=self.dtype).reshape(b.shape)

    def free(self):
        self._lu = None


class SciPyBatchSolver:
    """CPU drop-in for CuDSSBatchSolver (loop of SuperLU factorizations)."""

    def __init__(self, crow, col, n, n_rhs, nbatch, device="cpu",
                 dtype=torch.float64):
        self.n, self.n_rhs, self.nbatch = n, n_rhs, nbatch
        self.device, self.dtype = torch.device("cpu"), dtype
        self.crow = crow.to("cpu")
        self.col = col.to("cpu")
        self._single = [SciPySolver(crow, col, n, n_rhs, dtype=dtype)
                        for _ in range(nbatch)]
        counts = self.crow[1:] - self.crow[:-1]
        self.row = torch.repeat_interleave(torch.arange(n), counts)
        self.stamp = 0

    def factorize(self, values):
        for i in range(self.nbatch):
            self._single[i].factorize(values[i])
        self.stamp += 1

    def solve(self, b):
        shared = b.dim() == 2
        return torch.stack([s.solve(b if shared else b[i])
                            for i, s in enumerate(self._single)])

    def free(self):
        for s in self._single:
            s.free()


def _default_mtype(dtype, matrix_type):
    if matrix_type is not None:
        return matrix_type
    return "symmetric" if dtype.is_complex else "spd"


def make_solver(crow, col, n, n_rhs, device="cuda", dtype=torch.float64,
                matrix_type=None):
    """Device-dispatching factory: cuDSS on cuda, SuperLU on cpu."""
    mt = _default_mtype(dtype, matrix_type)
    if torch.device(device).type == "cuda":
        return CuDSSSolver(crow, col, n, n_rhs, device=device, dtype=dtype,
                           matrix_type=mt)
    return SciPySolver(crow, col, n, n_rhs, dtype=dtype)


def make_batch_solver(crow, col, n, n_rhs, nbatch, device="cuda",
                      dtype=torch.float64, matrix_type=None):
    mt = _default_mtype(dtype, matrix_type)
    if torch.device(device).type == "cuda":
        return CuDSSBatchSolver(crow, col, n, n_rhs, nbatch, device=device,
                                dtype=dtype, matrix_type=mt)
    return SciPyBatchSolver(crow, col, n, n_rhs, nbatch, dtype=dtype)
