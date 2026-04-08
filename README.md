# SimPEG DC Resistivity with PyTorch Backend

A PyTorch implementation of SimPEG's DC resistivity simulations with automatic differentiation.

## Overview

This repository provides PyTorch-enabled versions of SimPEG's core modules, specifically optimized for DC resistivity simulations. The implementation leverages PyTorch's automatic differentiation capabilities.

## Features

- PyTorch backend for SimPEG DC resistivity simulations in 2.5D and 3D.
- GPU support
- Automatic differentiation for gradient-based inversions
- Custom solvers optimized for sparse linear systems
- Selective replacement of discretize modules while preserving compiled extensions
- Compatible with existing SimPEG workflows

## Requirements

- Python 3.10
- CUDA-capable GPU (optional)
- Anaconda or Miniconda

## Installation

### Step 1: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate simpeg-pytorch
```

### Step 2: Install SimPEG

```bash
pip install simpeg==0.18.1
```

### Step 3: Install PyTorch

Choose the appropriate PyTorch installation for your system:

**For CUDA 12.4 (GPU acceleration):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**For CPU only:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

**For other CUDA versions, see:** https://pytorch.org/get-started/locally/

### Step 4: Install pydiso

```bash
conda install -c conda-forge pydiso
```

### Step 5: Install PyTorch Modifications

```bash
python install.py
```

### Verification

Test the installation:

```python
import SimPEG
from SimPEG.config import SimpegConfig

# Check if PyTorch backend is active
cfg = SimpegConfig()
print(f"PyTorch backend active: {cfg.torch_is_active}")

# Test basic functionality
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Quick Start

```bash
# 1. Create and activate environment
conda env create -f environment.yml
conda activate simpeg-pytorch

# 2. Install SimPEG
pip install simpeg==0.18.1

# 3. Install PyTorch (choose GPU or CPU version)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Install pydiso
conda install -c conda-forge pydiso

# 5. Install PyTorch modifications
python install.py
```

## Architecture

The implementation modifies the following SimPEG components:

- **SimPEG Core**: Complete replacement with PyTorch tensor operations
- **Discretize**: Selective replacement preserving compiled extensions
- **Custom Solvers**: Four solver backends for the linear systems arising from DC resistivity (see below)
- **Utilities**: Enhanced simulation utilities in `utils/` module

## Solvers

The repository includes four solvers, each implemented as a `torch.autograd.Function` so that gradients flow through the linear solve:

| Solver | Device | Strategy | Backward pass |
|--------|--------|----------|---------------|
| **SuperLU** | CPU | Sparse LU factorization (`scipy.sparse.linalg.splu`) | Conjugate-transpose solve reusing LU factors |
| **Pardiso** | CPU | Intel MKL Pardiso via `pymatsolver` | Reuses existing LDL^T factorization with `transpose=True` (no refactorization of A^T) |
| **PCG-GPU** | CUDA | Jacobi-preconditioned Conjugate Gradient with `torch.sparse.mm` | Same PCG (A is SPD, so A^T = A) |
| **Dense-GPU** | CUDA | Densifies A and solves with `torch.linalg.solve` | Native PyTorch autograd through `torch.linalg.solve` |

### Solver routing

Routing is handled automatically by `SolverWrapD` based on the `SimpegConfig` singleton:

- `device="cpu"` + `solver="superlu"` -> **SuperLU**
- `device="cpu"` + `solver="pardiso"` -> **Pardiso**
- `device="cuda"` -> **PCG-GPU** (via `SolverWrapD`) or **Dense-GPU** (manual integration in benchmarks)

## Benchmark: 2.5D DC Resistivity (forward + backward)

Total time (forward + backward) relative to SuperLU, measured on 2D resistivity models with `nky=11` wavenumbers, dipole-dipole survey:

| Params | SuperLU | Pardiso | PCG-GPU | Dense-GPU |
|-------:|--------:|--------:|--------:|----------:|
| 100 | 1.00x | 0.85x | 0.56x | 1.72x |
| 500 | 1.00x | 0.90x | 0.76x | 1.27x |
| 1000 | 1.00x | 0.97x | 0.64x | 1.10x |
| 2010 | 1.00x | 0.97x | 0.46x | 0.58x |
| 3000 | 1.00x | 1.07x | 0.33x | 0.39x |
| 4000 | 1.00x | 1.06x | 0.23x | 0.25x |
| 5000 | 1.00x | 1.08x | N/A | N/A |
| 6000 | 1.00x | 1.08x | N/A | N/A |
| 8000 | 1.00x | 1.07x | N/A | N/A |
| 9000 | 1.00x | 1.11x | N/A | N/A |
| 10000 | 1.00x | 1.04x | N/A | N/A |

> Values > 1.0 mean faster than SuperLU; values < 1.0 mean slower.

### Analysis

- **SuperLU** is the most robust baseline for 2D problems. Sparse LU factorization scales well and has minimal overhead per wavenumber.
- **Pardiso** matches SuperLU closely. Its LDL^T backward reuse gives a significant speedup on the backward pass alone (6x-27x), but the overall forward+backward time is similar because the forward solve is comparable.
- **PCG-GPU** is slower than CPU solvers for 2D problems. The iterative CG runs once per wavenumber (11 solves), and the per-iteration overhead of sparse-dense GPU operations dominates for the relatively small systems that 2D meshes produce. For large 3D systems (nC > 50k), GPU parallelism is expected to dominate.
- **Dense-GPU** works only for small meshes. Densifying the sparse system matrix causes VRAM to grow as O(n^2), making it infeasible beyond ~4000 parameters on a typical 8GB GPU. When it fits in memory, `torch.linalg.solve` is competitive for very small systems but quickly falls behind.

### Other solvers worth exploring (not implemented here)

There are several GPU-native sparse direct and iterative solvers that could outperform the options above for large 3D problems:

- **cuSOLVER** (`cusolverSpcsrlsvlu`, `cusolverSpcsrlsvchol`): NVIDIA's sparse direct solvers, accessible via CuPy. Sparse LU/Cholesky directly on GPU without densification.
- **CHOLMOD on GPU**: Sparse Cholesky factorization with GPU acceleration (SuiteSparse). Ideal for SPD systems like DC resistivity.
- **AMGX**: NVIDIA's algebraic multigrid solver. Excellent for large-scale elliptic PDEs, which is exactly what DC resistivity produces.
- **PETSc + GPU**: Distributed sparse solvers with GPU backends (CUDA, HIP). Overkill for single-GPU but powerful for multi-GPU clusters.
- **Sparse QR on GPU**: For non-symmetric systems or least-squares formulations.

These solvers are not integrated in this repository because the focus is on demonstrating differentiable DC resistivity with PyTorch autograd. The solver interface (`torch.autograd.Function` with `forward`/`backward`) is modular enough that any of the above could be plugged in following the same pattern as `SuperLUBatch` or `PardisoBatch`.

## Examples

See the `examples/` directory for complete working examples:

- `fwd_dcr_plane_2d.ipynb`: DC resistivity forward modeling in 2D plane geometry
- `fwd_dcr_topo_2d.ipynb`: DC resistivity forward modeling with 2D topography
- `fwd_dcr_topo_3d.ipynb`: DC resistivity forward modeling with 3D topography
- `benchmark_solvers_dc.py`: Full benchmark script (generates plots + Excel export)

## Performance

- Exact gradients via autograd, replacing SimPEG's finite-difference Jacobian approximation for DC 2.5D and 3D.
- Memory-optimized forward: solver factorizations are released after the forward pass when using autograd (backward is handled by the computation graph, not by stored Ainv objects).

## Limitations

- Currently supports DC resistivity simulations only
- GPU solvers (PCG-GPU, Dense-GPU) are not competitive for 2D problems due to small system sizes; expected advantage for 3D with nC > 50k
- Some advanced SimPEG features may not be fully compatible because this development was based on SimPEG 0.18.1

## Contributing

This is research extension under active development. Please report issues or contribute improvements via GitHub.

## Citation

If you use this extension in your research, please cite:

```bibtex
@extension{simpeg_dc_pytorch,
  title = {SimPEG DC Resistivity with PyTorch Backend},
  author = {Edwin Cuellar},
  year = {2025},
  url = {https://github.com/eacuellarq/simpeg-dc-pytorch},
  note = {Research extension for PyTorch geophysical simulations}
}
```

## Related Publications

- [Add relevant papers here when published]

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This work builds upon the SimPEG framework:

- Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., & Oldenburg, D. W. (2015). SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications. Computers & Geosciences, 85, 142-154.

## Support
For questions and support:

- Open an issue on GitHub
- Contact: eaqm1228@hotmail.com

