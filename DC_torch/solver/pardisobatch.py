
import os
import torch
import numpy as np
from scipy.sparse import csr_matrix, random
from scipy.sparse.linalg import splu
import time
from pymatsolver import Pardiso
import psutil  # Para monitorear memoria

# Importar las implementaciones
from solver.superLUbatch import SuperLUBatch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Copiar las clases PardisoBatch y PardisoBatchAlt del código original
class PardisoBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        assert A.layout == torch.sparse_coo, "A must be a COO sparse tensor"
        assert A.device == torch.device('cpu'),   "A must be on CPU"
        assert B.device == torch.device('cpu'),   "B must be on CPU"
        assert A.size(0) == B.size(0),            "Incompatible dimensions"
        
        A_coo = A.coalesce()
        rows_np = A_coo.indices()[0].numpy()
        cols_np = A_coo.indices()[1].numpy()
        values_np = A_coo.values().numpy()
        shape = A_coo.shape
        
        # Create CSR matrix for Pardiso
        csr_A = csr_matrix((values_np, (rows_np, cols_np)), shape=shape)
        
        # Create Pardiso solver
        solver = Pardiso(csr_A)
        
        B_np = B.detach().numpy()
        X_np = solver * B_np
        X = torch.from_numpy(X_np).to(dtype=B.dtype)
        
        # Save everything for backward
        ctx.solver = solver
        ctx.rows_np = rows_np
        ctx.cols_np = cols_np
        ctx.X = X
        ctx.shape = shape
        
        return X

    @staticmethod
    def backward(ctx, grad_output):
        solver = ctx.solver
        rows_np = ctx.rows_np
        cols_np = ctx.cols_np
        X = ctx.X
        shape = ctx.shape
        
        go_np = grad_output.detach().numpy()
        
        # Get the original matrix in transposed form
        csr_A_T = solver.A.T.tocsr()
        solver_T = Pardiso(csr_A_T)
        
        # Solve the conjugate-transposed system
        grad_B_np = solver_T * go_np
        grad_B = torch.from_numpy(grad_B_np).to(dtype=grad_output.dtype)
        
        # Gradient with respect to A
        rows = torch.from_numpy(rows_np).long()
        cols = torch.from_numpy(cols_np).long()
        grad_B_rows = grad_B[rows]               # [nnz, batch]
        X_cols_conj = X[cols].conj()             # [nnz, batch]
        grad_A_vals = -(grad_B_rows * X_cols_conj).sum(dim=1)
        indices = torch.stack([rows, cols], dim=0)
        grad_A = torch.sparse_coo_tensor(indices, grad_A_vals, shape)
        
        return grad_A, grad_B