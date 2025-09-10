import time
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

class SuperLUBatch(torch.autograd.Function):
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
        
        csr_A = csr_matrix((values_np, (rows_np, cols_np)), shape=shape)
        csc_A = csr_A.tocsc()
        invA = splu(csc_A)
        
        B_np = B.detach().numpy()
        X_np = invA.solve(B_np)
        X = torch.from_numpy(X_np).to(dtype=B.dtype)
        
        # Save everything for backward
        ctx.invA = invA
        ctx.rows_np = rows_np
        ctx.cols_np = cols_np
        ctx.X = X
        ctx.shape = shape
        
        return X

    @staticmethod
    def backward(ctx, grad_output):
        invA = ctx.invA
        rows_np = ctx.rows_np
        cols_np = ctx.cols_np
        X = ctx.X
        shape = ctx.shape
        
        go_np = grad_output.detach().numpy()
        # Solve the conjugate-transposed system
        grad_B_np = invA.solve(go_np, trans='H')
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
