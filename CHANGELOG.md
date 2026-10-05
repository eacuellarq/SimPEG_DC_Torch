# Changelog

## 0.2.0 — 5 October 2026

A different engine under the same repository name. Version 0.1.0 was the
companion code for **ADERT** (Rincon et al., 2026, *Journal of Applied
Geophysics* 252, 106365); it patched modules inside an installed SimPEG 0.18.
This release replaces it.

### The change that produced the others

SimPEG is no longer modified. `dctorch` imports stock SimPEG 0.25, extracts the
linear maps it needs at build time, and re-validates them against SimPEG's own
operators on every construction. Torch never crosses into `simpeg` or
`discretize`, so the host library keeps its compiled extensions and its numpy
arrays, and an upgrade is no longer a merge.

### Added

- **Exact adjoint sensitivities that are never assembled.** `Jtvec` is the
  reverse pass of the forward solve and costs about one extra simulation, so
  peak memory follows the simulation rather than the number of measurements.
- **A batched adjoint** (`TorchDC2D.Jmatrix`) for where the matrix is wanted
  anyway: its right-hand side is the receiver projection transposed and is
  known before the run, so every datum's adjoint solve is one batched solve
  against the factorization the forward already computed — 0.31 s against
  30.0 s for a stored Jacobian on a 661-measurement line.
- **cuDSS on the GPU** with the symbolic phase cached per sparsity pattern,
  SuperLU on the CPU, behind one interface.
- **Single precision** as well as double; single is the useful default.
- **Induced polarization**: `TorchIP2D`, `TorchIP3D`, a joint non-linear
  `TorchDCIP2D`, and `TorchCR2D` for complex resistivity.
- **Two optimizers**, both exposed as SimPEG `Optimization`s so the stock
  directives drive them: `TorchLBFGS`, and `TorchGaussNewton` with an Occam
  beta search that chooses the trade-off parameter inside each step.
- **Wavenumbers fitted over the survey's electrode distances** with
  non-negative weights, rather than over the mesh.
- **Singularity removal** in 2.5-D, off by default.
- **A test suite in CI**: forward parity 7e-16 to 2e-15 against stock SimPEG
  and gradients to 1e-7 against finite differences, over {2.5-D, 3-D} x
  {tensor, octree} x {GPU, CPU}.

### Removed

- `DC_torch/`, the patched copies of SimPEG and discretize modules. They remain
  in the history at `96b4e46` for anyone who needs the ADERT-era code.

### Not included

The engine was developed against two field campaigns whose data are
proprietary. Their scripts are not part of this release, and no published
result depends on them: the benchmarks here use SimPEG's published tutorial,
the Century line distributed with the Transform 2020 tutorial, a field-scale
synthetic generated from fixed seeds, and AD-TLERT's synthetics from its MIT
repository.
