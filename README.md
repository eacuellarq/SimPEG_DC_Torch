# SimPEG DC Resistivity with PyTorch Backend

A PyTorch implementation of SimPEG's DC resistivity simulations with automatic differentiation.

## Overview

This repository provides PyTorch-enabled versions of SimPEG's core modules, specifically optimized for DC resistivity simulations. The implementation leverages PyTorch's automatic differentiation capabilities.

> ### 📄 Associated publication
> This software accompanies the following peer-reviewed article. **If you use this code, please cite the paper:**
>
> Rincón, F., Aleardi, M., Cuellar, E., Berti, S., Tognarelli, A., & Stucchi, E. —
> "ADERT: Automatic differentiation-based electrical resistivity tomography inversion",
> *Journal of Applied Geophysics*, **2026**, article 106365.
> DOI: [10.1016/j.jappgeo.2026.106365](https://doi.org/10.1016/j.jappgeo.2026.106365) ·
> [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926985126002740)

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

> `pydiso` is only required for the **Pardiso** CPU solver. If you only plan to use
> SuperLU (CPU) or the GPU solvers, you can skip this step.

### Step 5 (optional): CuPy for the experimental GPU sparse solver

The experimental `SpSolverGPU` backend (see [Solvers](#solvers)) relies on CuPy. It is
**not** required for the default GPU path (PCG-GPU). Install only if you want to use it:

```bash
pip install cupy-cuda12x   # match your CUDA toolkit
```

### Step 6: Install PyTorch Modifications

```bash
python install.py
```

This overwrites the installed `SimPEG`/`discretize` packages with the PyTorch-enabled
versions in `DC_torch/`, and installs `solver/` and `utils/` as top-level packages in
`site-packages` (so they are importable as `from solver...` and `from utils...`).

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

# 4. Install pydiso (only needed for the Pardiso solver)
conda install -c conda-forge pydiso

# 5. Install PyTorch modifications
python install.py
```

## Usage

The PyTorch backend is controlled through the `SimpegConfig` singleton. You activate it,
choose a device and a solver, and then build and run a `SimulationDCResistivity` exactly
as you would in standard SimPEG. Because the forward solve is differentiable, gradients
of any scalar built from the predicted data flow back to the model through `.backward()`.

```python
import torch
from SimPEG.config import SimpegConfig
from utils.simulation_dc_resistivity import SimulationDCResistivity

# 1. Activate the PyTorch backend and select device + solver
cfg = SimpegConfig()
cfg.torch_is_active = True
cfg.dtype = torch.float64
cfg.device = "cpu"        # "cpu" or "cuda"
cfg.solver = "superlu"    # see "Solver routing" below

# 2. Differentiable resistivity model (requires_grad=True enables autograd)
m = torch.tensor(model_np, requires_grad=True, dtype=torch.float64)

# 3. Survey definition
surveyinfo = dict(
    nElec=24, sep=2.0, pi=0, depth=float(nz),
    typ_survey="dipole-dipole", dim="2D",
    data_type="volt", nlines=6, pf=48.0,
)

# 4. Build the simulation
sim = SimulationDCResistivity(model=m, surveyinfo=surveyinfo, tensor=True)
sim.initialize_survey()
sim.initialize_mesh(adjust_model=False, m_background=torch.mean(m))

# 5. Forward modeling (2.5D uses nky wavenumbers)
dpred = sim.forward_modeling(nky=11, error=0.0, only_data=True)

# 6. Exact gradient via autograd (replaces SimPEG's finite-difference Jacobian)
dpred.sum().backward()
grad = m.grad   # dL/dm at every cell
```

> The `solver/` and `utils/` packages are installed at the top level of `site-packages`
> by `install.py`, so the imports above (`from solver...`, `from utils...`) work from any
> working directory. The benchmark script, however, imports `utils` and `solver` directly
> and is meant to be run from inside `DC_torch/` (see [Examples](#examples)).

For complete, runnable end-to-end workflows (mesh, survey, plotting, inversion), see the
notebooks in `examples/`.

## Architecture

The implementation modifies the following SimPEG components:

- **SimPEG Core**: Complete replacement with PyTorch tensor operations
- **Discretize**: Selective replacement preserving compiled extensions
- **Custom Solvers**: Differentiable solver backends for the linear systems arising from DC resistivity (see below)
- **Utilities**: Enhanced simulation utilities in `utils/` module

## Solvers

The repository includes four production solver backends, each implemented as a
`torch.autograd.Function` so that gradients flow through the linear solve:

| Solver | Device | Strategy | Backward pass |
|--------|--------|----------|---------------|
| **SuperLU** | CPU | Sparse LU factorization (`scipy.sparse.linalg.splu`) | Conjugate-transpose solve reusing LU factors |
| **Pardiso** | CPU | Intel MKL Pardiso via `pymatsolver` | Reuses existing LDL^T factorization with `transpose=True` (no refactorization of A^T) |
| **PCG-GPU** | CUDA | Jacobi-preconditioned Conjugate Gradient with `torch.sparse.mm` | Same PCG (A is SPD, so A^T = A) |
| **Dense-GPU** | CUDA | Densifies A and solves with `torch.linalg.solve` | Native PyTorch autograd through `torch.linalg.solve` |

In addition, an **experimental** `SpSolverGPU` backend (`solver/spsolverGPU.py`) wraps
CuPy's sparse direct solver (`cupyx.scipy.sparse.linalg.spsolve`, i.e. cuSOLVER) as a
`torch.autograd.Function`. It performs a true sparse solve on the GPU without densifying
A. It is **not wired into `SolverWrapD` routing** and requires CuPy; treat it as a
prototype/reference rather than a selectable option.

### Solver routing

Routing is handled automatically by `SolverWrapD` based on the `SimpegConfig` singleton.
Note that on CUDA the device takes precedence: any solver other than `dense_gpu` falls
through to PCG-GPU.

- `device="cpu"` + `solver="superlu"` -> **SuperLU**
- `device="cpu"` + `solver="pardiso"` -> **Pardiso**
- `device="cuda"` + `solver="dense_gpu"` -> **Dense-GPU**
- `device="cuda"` + any other solver -> **PCG-GPU**

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

### Other solvers worth exploring

There are several GPU-native sparse direct and iterative solvers that could outperform the options above for large 3D problems:

- **cuSOLVER** (`cusolverSpcsrlsvlu`, `cusolverSpcsrlsvchol`): NVIDIA's sparse direct solvers, accessible via CuPy. A reference implementation already exists in `solver/spsolverGPU.py` (`SpSolverGPU`), but it is not yet wired into the routing layer.
- **CHOLMOD on GPU**: Sparse Cholesky factorization with GPU acceleration (SuiteSparse). Ideal for SPD systems like DC resistivity.
- **AMGX**: NVIDIA's algebraic multigrid solver. Excellent for large-scale elliptic PDEs, which is exactly what DC resistivity produces.
- **PETSc + GPU**: Distributed sparse solvers with GPU backends (CUDA, HIP). Overkill for single-GPU but powerful for multi-GPU clusters.
- **Sparse QR on GPU**: For non-symmetric systems or least-squares formulations.

The solver interface (`torch.autograd.Function` with `forward`/`backward`) is modular enough that any of the above can be plugged in following the same pattern as `SuperLUBatch` or `PardisoBatch`.

## Examples

See the `examples/` directory for complete working examples:

- `fwd_dcr_plane_2d.ipynb`: DC resistivity forward modeling in 2D plane geometry
- `fwd_dcr_topo_2d.ipynb`: DC resistivity forward modeling with 2D topography
- `fwd_dcr_topo_3d.ipynb`: DC resistivity forward modeling with 3D topography
- `benchmark_solvers_dc.py`: Full benchmark script (generates plots + Excel export)

The benchmark imports `utils` and `solver` directly, so run it from inside `DC_torch/`:

```bash
cd DC_torch
python ../examples/benchmark_solvers_dc.py
```

## Performance

- Exact gradients via autograd, replacing SimPEG's finite-difference Jacobian approximation for DC 2.5D and 3D.
- Memory-optimized forward: solver factorizations are released after the forward pass when using autograd (backward is handled by the computation graph, not by stored Ainv objects).

## Limitations

- Currently supports DC resistivity simulations only
- GPU solvers (PCG-GPU, Dense-GPU) are not competitive for 2D problems due to small system sizes; expected advantage for 3D with nC > 50k
- Some advanced SimPEG features may not be fully compatible because this development was based on SimPEG 0.18.1

## Contributing

This is a research extension under active development. Please report issues or contribute improvements via GitHub.

## Citation

This software is associated with a peer-reviewed publication (see [Associated publication](#-associated-publication) above). **Please cite the paper as the primary reference.** If you additionally want to cite the software itself:

```bibtex
@software{simpeg_dc_pytorch,
  title  = {SimPEG DC Resistivity with PyTorch Backend},
  author = {Cuellar, Edwin},
  year   = {2025},
  url    = {https://github.com/eacuellarq/SimPEG_DC_Torch},
  note   = {Software accompanying doi:10.1016/j.jappgeo.2026.106365}
}
```

## Related Publications

```bibtex
@article{RINCON2026106365,
  author  = {Rincón, Felipe and Aleardi, Mattia and Cuellar, Edwin and Berti, Sean and Tognarelli, Andrea and Stucchi, Eusebio},
  title   = {ADERT: Automatic differentiation-based electrical resistivity tomography inversion},
  journal = {Journal of Applied Geophysics},
  year    = {2026},
  pages   = {106365},
  issn    = {0926-9851},
  doi     = {10.1016/j.jappgeo.2026.106365},
  url     = {https://www.sciencedirect.com/science/article/pii/S0926985126002740}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This work builds upon the SimPEG framework:

- Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., & Oldenburg, D. W. (2015). SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications. Computers & Geosciences, 85, 142-154.

## Support

For questions and support:

- Open an issue on GitHub
- Contact: eaqm1228@hotmail.com
