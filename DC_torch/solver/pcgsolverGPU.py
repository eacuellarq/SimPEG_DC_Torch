"""
PCGSolverGPU — Preconditioned Conjugate Gradient on GPU via torch.autograd
===========================================================================
Designed for DC resistivity: A = G^T M_sigma G + BC is symmetric positive
definite. Uses implicit differentiation for backward pass.

Why this beats Pardiso on GPU:
  - sparse mat-vec (torch.sparse.mm) is massively parallel on GPU
  - Jacobi-preconditioned CG converges in O(sqrt(kappa)) iterations
  - Each iteration is O(nnz) — GPU throughput dominates
  - No CPU<->GPU transfer overhead (everything stays on device)
  - Backward pass reuses same PCG (A is SPD so A^T = A)

Author: Edwin Cuellar (eacuellarq@eafit.edu.co)
"""

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sparse_diagonal(A):
    """Extract diagonal from sparse COO tensor without densifying."""
    A = A.coalesce()
    idx = A.indices()
    vals = A.values()
    mask = idx[0] == idx[1]
    diag = torch.zeros(A.shape[0], device=A.device, dtype=A.dtype)
    diag.scatter_add_(0, idx[0][mask], vals[mask])
    return diag


@torch.no_grad()
def _pcg_solve(A, B, M_inv, tol=1e-7, maxiter=5000):
    """
    Batched Preconditioned Conjugate Gradient for SPD system AX = B.

    Parameters
    ----------
    A : torch.sparse_coo_tensor (n, n), coalesced, on CUDA
    B : torch.Tensor (n,) or (n, k) — right-hand sides
    M_inv : torch.Tensor (n,) — inverse of diagonal preconditioner
    tol : float — relative residual tolerance
    maxiter : int — maximum CG iterations

    Returns
    -------
    X : torch.Tensor, same shape as B
    """
    single = B.dim() == 1
    if single:
        B = B.unsqueeze(1)

    n, k = B.shape
    device = B.device
    dtype = B.dtype

    X = torch.zeros(n, k, device=device, dtype=dtype)
    R = B.clone()
    Z = M_inv.unsqueeze(1) * R
    P = Z.clone()
    rz = (R * Z).sum(dim=0)                    # (k,)
    b_norm = B.norm(dim=0).clamp(min=1e-30)     # (k,)

    for it in range(maxiter):
        AP = torch.sparse.mm(A, P)             # (n, k) — the hot op
        pAp = (P * AP).sum(dim=0)              # (k,)
        alpha = rz / pAp.clamp(min=1e-30)      # (k,)

        X = X + alpha.unsqueeze(0) * P
        R = R - alpha.unsqueeze(0) * AP

        # Convergence check (worst column)
        r_norm = R.norm(dim=0)
        rel_res = (r_norm / b_norm).max().item()
        if rel_res < tol:
            break

        Z = M_inv.unsqueeze(1) * R
        rz_new = (R * Z).sum(dim=0)
        beta = rz_new / rz.clamp(min=1e-30)
        P = Z + beta.unsqueeze(0) * P
        rz = rz_new

    # Log convergence info
    if _pcg_solve._verbose:
        status = "converged" if rel_res < tol else "NOT CONVERGED"
        print(f"    PCG: {it+1} iters, rel_res={rel_res:.2e}, n={n}, k={k} [{status}]")

    if single:
        X = X.squeeze(1)
    return X

_pcg_solve._verbose = True  # Set to False to disable logging


# ---------------------------------------------------------------------------
# Autograd Function  (differentiable!)
# ---------------------------------------------------------------------------

class PCGSolverGPU(torch.autograd.Function):
    """
    GPU Preconditioned Conjugate Gradient for SPD sparse systems.

    Forward:  solve AX = B via Jacobi-PCG on CUDA
    Backward: implicit differentiation
              dL/dB = A^{-1} dL/dX   (A is SPD => A^T = A)
              dL/dA = -lambda @ X^T  evaluated at nonzero positions of A

    Interface: PCGSolverGPU.apply(A, B)
    Same signature as SuperLUBatch / PardisoBatch.
    """

    # Tunable class-level defaults
    tol = 1e-7
    maxiter = 5000

    @staticmethod
    def forward(ctx, A, B):
        assert A.layout == torch.sparse_coo, "A must be a COO sparse tensor"
        assert A.is_cuda, "A must be on CUDA"
        assert B.is_cuda, "B must be on CUDA"

        A = A.coalesce()

        # Jacobi preconditioner: M^{-1} = 1 / diag(A)
        diag = _sparse_diagonal(A)
        M_inv = 1.0 / diag.clamp(min=1e-30)

        X = _pcg_solve(A, B, M_inv, PCGSolverGPU.tol, PCGSolverGPU.maxiter)

        ctx.save_for_backward(A, X)
        ctx.M_inv = M_inv

        return X

    @staticmethod
    def backward(ctx, grad_output):
        A, X = ctx.saved_tensors
        M_inv = ctx.M_inv

        # Adjoint solve: A lambda = grad_output  (A is SPD => A^T = A)
        Lambda = _pcg_solve(A, grad_output, M_inv,
                            PCGSolverGPU.tol, PCGSolverGPU.maxiter)

        # Gradient w.r.t. A at nonzero positions: dL/dA_ij = -lambda_i * x_j
        # Memory-efficient: process column-by-column to avoid OOM on large nnz
        indices = A.indices()
        rows, cols = indices[0], indices[1]
        nnz = rows.shape[0]

        grad_A_vals = torch.zeros(nnz, device=A.device, dtype=A.dtype)
        if Lambda.dim() == 1:
            grad_A_vals = -(Lambda[rows] * X[cols].conj())
        else:
            # Process each RHS column to avoid allocating (nnz, k) tensors
            for j in range(Lambda.shape[1]):
                grad_A_vals -= Lambda[:, j][rows] * X[:, j][cols].conj()

        grad_A = torch.sparse_coo_tensor(
            indices, grad_A_vals, A.shape, device=A.device, dtype=A.dtype
        )

        return grad_A, Lambda
