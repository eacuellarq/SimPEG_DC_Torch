import torch
import cupy as cp
from cupyx.scipy.sparse import csc_matrix
from cupyx.scipy.sparse.linalg import spsolve

class SpSolverGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b):
        assert A.layout == torch.sparse_coo, "A must be COO sparse tensor"
        assert A.device.type == 'cuda', "A must be on CUDA"
        assert b.device.type == 'cuda', "b must be on CUDA"
        
        A = A.coalesce()
        indices = A.indices()
        values = A.values()
        shape = A.shape
        
        rows = indices[0].to(torch.int32)
        cols = indices[1].to(torch.int32)
        values_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(values))
        b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b))
        
        rows_cp = cp.asarray(rows.cpu())
        cols_cp = cp.asarray(cols.cpu())
        csc_A = csc_matrix((values_cp, (rows_cp, cols_cp)), shape=shape)
        
        x_cp = spsolve(csc_A, b_cp)
        x = torch.utils.dlpack.from_dlpack(x_cp.toDlpack())
        
        ctx.A_csc = csc_A
        ctx.x_best = x
        ctx.rows = rows
        ctx.cols = cols
        ctx.shape = shape
        
        return x

    @staticmethod
    def backward(ctx, grad_output):
        A_csc = ctx.A_csc
        x_best = ctx.x_best
        rows = ctx.rows
        cols = ctx.cols
        shape = ctx.shape
        
        grad_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(grad_output))
        grad_b_cp = spsolve(A_csc.T.conj(), grad_cp)
        grad_b = torch.utils.dlpack.from_dlpack(grad_b_cp.toDlpack())
        
        grad_b_rows = grad_b[rows]
        x_cols_conj = x_best[cols].conj()
        grad_A_vals = -grad_b_rows * x_cols_conj
        
        indices = torch.stack([rows, cols], dim=0)
        grad_A = torch.sparse_coo_tensor(indices, grad_A_vals, shape, device=grad_b.device)
        
        return grad_A, grad_b