# DC_torch — PyTorch-modified SimPEG components

This directory holds the PyTorch-enabled sources that `install.py` (in the repository
root) deploys into your active conda environment. You normally do not edit anything here
at install time; the developer instruction of "place your modules here" applies only if
you are extending the backend.

## Contents

| Path | Installed to | Purpose |
|------|--------------|---------|
| `SimPEG/` | overwrites the installed `SimPEG` package | Complete PyTorch replacement of the SimPEG core (tensorized operators, `config.py` with the `SimpegConfig` singleton, DC resistivity simulation, differentiable `SolverWrapD`). |
| `discretize/` | selectively overwrites the installed `discretize` package | PyTorch-aware mesh/operator code. Only files that already exist in the vanilla package are replaced, so compiled extensions (e.g. `.so`/`.pyd`) are preserved. |
| `solver/` | installed as a top-level `solver` package in `site-packages` | Differentiable linear solvers (see below). Imported as `from solver.superLUbatch import SuperLUBatch`, etc. |
| `utils/` | installed as a top-level `utils` package in `site-packages` | High-level helpers, including `SimulationDCResistivity` and inversion result tracking. Imported as `from utils.simulation_dc_resistivity import SimulationDCResistivity`. |

## Solvers (`solver/`)

Each solver is a `torch.autograd.Function`, so gradients propagate through the linear
solve:

- `superLUbatch.py` — **SuperLUBatch** (CPU): sparse LU via `scipy.sparse.linalg.splu`; backward reuses the factorization with a conjugate-transpose solve.
- `pardisobatch.py` — **PardisoBatch** (CPU): Intel MKL Pardiso via `pymatsolver`/`pydiso`; backward reuses the LDL^T factorization with `transpose=True`.
- `pcgsolverGPU.py` — **PCGSolverGPU** (CUDA): Jacobi-preconditioned Conjugate Gradient with `torch.sparse.mm`; backward solves the adjoint system (A is SPD, so A^T = A).
- `spsolverGPU.py` — **SpSolverGPU** (CUDA, experimental): sparse GPU direct solve via CuPy (`cupyx.scipy.sparse.linalg.spsolve`, cuSOLVER). Requires CuPy and is **not** wired into `SolverWrapD` routing.

Routing between SuperLU / Pardiso / PCG-GPU / Dense-GPU is driven by the `device` and
`solver` attributes of `SimpegConfig`. See the root `README.md` ("Solver routing") for
details.

## Installation

Do not run anything from inside this directory. From the repository root:

```bash
python install.py
```

To restore the original packages:

```bash
conda install --force-reinstall simpeg=0.18.1
```
