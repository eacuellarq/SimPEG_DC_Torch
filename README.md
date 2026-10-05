# dctorch — DC resistivity and IP on the GPU, with exact adjoint sensitivities

`dctorch` solves the DC-resistivity forward problem on a graphics card and
differentiates it exactly, so an inversion is driven by the gradient of its own
objective function rather than by a stored table of sensitivities. The
sensitivities are never assembled: `J^T v` is the reverse pass of the forward
solve, which costs about one extra simulation. Peak memory is therefore set by
the simulation, not by how many measurements were collected, and densifying a
survey no longer threatens the run.

It **extends stock SimPEG from the outside**. Nothing in `site-packages` is
patched or replaced: SimPEG and `discretize` stay on numpy and keep their
compiled extensions, torch lives entirely inside this package, and the problem
is still defined the SimPEG way — a `TensorMesh` or `TreeMesh`, a
`dc.survey.Survey`, the same boundary conditions. At build time the engine runs
SimPEG's own assembly once to extract a sparse linear map from conductivity to
the values of the system matrix, and validates that map against `sim.getA()`
before using it.

## What is in it

| | |
|---|---|
| DC resistivity | `TorchDC2D` (2.5-D, multi-wavenumber), `TorchDC3D` |
| Induced polarization | `TorchIP2D`, `TorchIP3D` (linearized), `TorchDCIP2D` (joint non-linear), `TorchCR2D` (complex resistivity) |
| Meshes | `TensorMesh` and `TreeMesh` (octree), with active cells |
| Linear solver | cuDSS on the GPU, SuperLU on the CPU — the same interface, dispatched on the device |
| Precision | float64 and float32 (float32 is the useful default: half the memory, about half again as fast) |
| Optimization | `TorchGaussNewton` (2.5-D) and `TorchLBFGS`, both exposed as SimPEG `Optimization`s, so `BetaSchedule`, `TargetMisfit`, `UpdateIRLS` and the other directives run unchanged |
| Robustness | `StudentTDataMisfit` for a heavy-tailed misfit |
| Accuracy option | singularity removal in 2.5-D (`singularity_removal=True`, off by default; see `notes/HANDOFF_2026-09-27.md` for where it pays and where it does not) |
| Wavenumbers | Fitted over the survey's electrode distances with positive weights, 12 by default (`quadrature="electrodes"`, as RES2DINV fits its own). On a 661-measurement field line its forward error against a 31-point reference is 0.0007 % median (SimPEG's own 12: 0.035 %), same inverted model. Fewer points are not equivalent on a real mesh (7 reach 2.5 % forward error), so 12 stays. `quadrature="simpeg"` gives stock SimPEG's points and bit parity; `DCTORCH_QUADRATURE=simpeg` restores that globally, to reproduce results made before this default |

Measured on the regression case in `dctorch/tests/test_simulation2d.py` (2.5-D,
5 000 cells, 21 electrodes, `nky=11`), on an RTX 4070 Laptop (8 GB):

```
[parity]   dpred torch vs stock SimPEG: rel-err 8.2e-16
[gradient] directional vs finite differences: 5.8e-08
[time]     eval 9.5 ms | eval+grad 18.5 ms | stock CPU eval 352 ms (37x)
```

The same test on the CPU path gives `9.5e-16` parity and `1.5x`. Field-scale
figures, the inversion races and the negative results are in `paper/NUMBERS.md`
and reproducible from `paper/benchmarks/`.

## Install

The CPU stack is complete — everything imports and every test passes on it.

```bash
conda env create -f environment.yml
conda activate dctorch
pip install -e .
```

For the GPU path add the two CUDA pieces, which are not on conda-forge:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install "nvmath-python[cu12]"
```

`nvmath-python` is the cuDSS binding and is CUDA-only. Without it `dctorch`
still imports and runs; asking for a `cuda` solver then raises with that reason.

Certified environment: Python 3.11, numpy 2.4, scipy 1.17, torch 2.11+cu128,
SimPEG 0.25.2, discretize 0.12, nvmath 1.0, CUDA 12.8.

## Use

The model is `m = log(rho)` on the full mesh, matching a stock simulation built
with `rhoMap=maps.ExpMap(mesh)`.

```python
import torch
from dctorch import TorchDC2D

eng = TorchDC2D(mesh, survey, nky=11, device="cuda")   # mesh/survey: plain SimPEG

m = torch.tensor(m0, device="cuda", requires_grad=True)
eng.misfit(m, dobs, w).backward()      # w * (dpred - dobs), squared
g = m.grad                             # exact J^T v, no Jacobian anywhere
```

The same reverse pass is available with any data-space vector in place of the
residual, which is what posterior analysis consumes:

```python
g = eng.Jtvec(m, v)          # J^T v, one extra simulation, no nD x nM array
```

A rank-k sketch of the sensitivities is k of those; the explicit Jacobian is nD
of them and is only affordable on small surveys.

To invert, hand the engine to an optimizer and build the inverse problem with
stock SimPEG:

```python
from dctorch import TorchGaussNewton            # or TorchLBFGS
opt = TorchGaussNewton(engine=eng, maxIter=25)
# regularization: stock SimPEG; directives: TargetMisfit only --
# beta is chosen inside every step (Occam), so no BetaSchedule
```

The defaults are the method validated on a field line (RES2DINV-style: the
program chooses, the user does not tune): 12 fitted wavenumbers in the
engine, and Gauss-Newton with the Occam beta search, which takes the largest
beta whose linearized misfit reaches the target. On that line, with the stock
recipe (golden sigmas, alpha_s = 1e-3), it reproduces stock SimPEG's
Gauss-Newton section (corr 0.992, near surface and at depth) in 4.6 s in
float64 and 2.8 s with `dtype=torch.float32` (same model, corr 1.0000),
against 204 s for stock SimPEG. `beta_search=False` restores one step per
beta level driven by a `BetaSchedule`.

The data error model matters more than any setting: the golden sigmas
(floor + 5 % |d|) reproduce stock SimPEG; a pure 5 % relative error on log
data weights the smallest, deepest voltages fully and changes the deep model
(corr 0.89 below 150 m) whatever the engine.

`TorchGaussNewton` takes one Gauss-Newton step per iteration on the exact
sensitivities: one J per step (the batched adjoint `Jmatrix` when it fits
`explicit_max_gib`, otherwise a `linearize`d forward with `Jvec`/`Jtvec`)
serves both the gradient and the normal equations, which are solved by
truncated CG (`cg_rtol`, default 1e-2) with a Jacobi preconditioner from the
exact diagonal of J^T D J. Steps are capped at `max_step` in log-resistivity
and accepted by an Armijo line search along a parabola built from f(0), the
slope and f(1). `inner="direct"` solves the step exactly in data space
(Woodbury: an nD x nD system), which on the line below converged no faster
than the truncated CG.

On the comparison line of `examples/inversion_2d_dctorch_compare.ipynb`
(8,816 cells, 195 data, same beta0 and directives;
`paper/benchmarks/scripts/gauss_newton_2d.py`), seeds 1-3, RTX 4070 Laptop:

| | time | rms log10 rho vs truth | SSIM |
|---|---|---|---|
| stock SimPEG `InexactGaussNewton`, CPU | 39.6 s | 0.350 | 0.401 |
| `TorchLBFGS` | 12.0-13.1 s | 0.347-0.364 | 0.341-0.357 |
| `TorchGaussNewton` (explicit J) | 2.5-2.7 s | 0.304-0.331 | 0.385-0.434 |

Memory: the explicit J is streamed from one linearization a few rows at a
time, and the matrix-free products never copy the assembly map, so
Gauss-Newton peaks no higher than `TorchLBFGS` (GPU above baseline, same
problem refined 2x: 652 MiB for both explicit GN and L-BFGS at 35k cells,
586 MiB matrix-free; 174 vs 302 MiB at 8.8k cells) while running 3.8-4.8x
faster in the explicit mode.

Gauss-Newton tends to overshoot the target misfit (chi2 0.41-0.63 against
L-BFGS's 0.75-0.96), since `TargetMisfit` checks after a whole step. It
needs the plain formulation: under singularity removal use `TorchLBFGS`.

The Occam search (`beta_search=True`, the default) chooses beta inside every step (Occam's inversion,
RES2DINV's "combined Marquardt and Occam"): with J fixed, the exact step for
any beta is one nD x nD Cholesky, so candidates are cheap, and the largest
beta whose linearized misfit reaches the target -- the smoothest model that
fits -- is taken. Drop `BetaSchedule` from the directives. On the line above it
ends at chi2 0.86-0.99 instead of overshooting, usually in fewer iterations
(8-10 against 11-16), and its final beta no longer depends on beta0: started
10^4 apart, two runs end within 8 % of each other. It returns the smoothest
model consistent with the noise, so on a blocky synthetic its rms against the
truth is ~0.02-0.03 higher than an overshooting run, at equal or better SSIM.
Needs the explicit J and `alpha_s > 0`.

Worked examples, including the two field cases and the comparisons against the
conventional workflow, are in `examples/`.

## Tests

```bash
python -m dctorch.tests.test_suite           # {2.5D,3D} x {Tensor,Tree} x {cuda,cpu}
python -m dctorch.tests.test_simulation2d    # the 2.5-D regression: parity + gradient
python -m dctorch.tests.test_ip2d            # and test_ip3d, test_cr2d, test_dcip2d
python -m dctorch.tests.test_linearization   # J v, J^T v, diag(J^T D J) vs Jmatrix
python -m dctorch.tests.test_gauss_newton    # GN pieces vs autograd, then inversions
python -m dctorch.tests.test_quadrature      # wavenumber fit vs SimPEG's, half-space accuracy
```

Those run with or without a GPU: the device sweep collapses to the CPU path when
`torch.cuda.is_available()` is false, and the parity and gradient assertions do
not depend on the device. `test_solver`, `test_protocol2d` and `test_singularity`
need a GPU. Continuous integration runs the first group on CPU.

Every engine also self-validates at construction: the extracted assembly map is
checked against SimPEG's own `getA` at two models, to 1e-10, on any mesh.

## Layout

```
dctorch/            the package: solver, simulations, IP/CR, optimization, tests
examples/           notebooks and scripts, including the field cases
paper/              the manuscript, its measured numbers, and the benchmark suite
paper/edits/        how the manuscript is edited, and the conventions it follows
notes/              session records: what was measured, what failed, and why
```

## Relation to ADERT

This engine is an improved version of the one that accompanied **ADERT**:
Rincon, Aleardi, Cuellar, Berti, Tognarelli and Stucchi (2026), *ADERT:
Automatic differentiation-based electrical resistivity tomography inversion*,
Journal of Applied Geophysics 252, 106365. That paper — which this author
co-wrote — showed that reverse-mode automatic differentiation gives exact ERT
sensitivities on a graphics card, and it remains the right citation for that
result. It is not the citation for this engine.

**Where to find ADERT**

| | |
|---|---|
| the paper | [doi:10.1016/j.jappgeo.2026.106365](https://doi.org/10.1016/j.jappgeo.2026.106365) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0926985126002740) |
| its companion code | the earlier state of this repository, at [github.com/eacuellarq/simpeg-dc-pytorch](https://github.com/eacuellarq/simpeg-dc-pytorch) before this rewrite |
| what it also covers | annealed Stein variational inference and DCT model compression for probabilistic ERT, which this engine does not reimplement |

**What this version improves.** It shares the idea and almost none of the
code, and one difference produced all the others: where SimPEG sits. The
earlier version replaced modules inside an installed SimPEG, which pinned it
to one old release, put the autograd graph inside a library that knew nothing
about torch, and made every upgrade a merge. This one imports stock SimPEG and
extends it from outside: torch never crosses into `simpeg` or `discretize`,
and the operators it extracts are re-validated against SimPEG's own on every
construction, so a wrong extraction fails loudly instead of quietly returning
a plausible number.

| | earlier version | this one |
|---|---|---|
| SimPEG | 0.18.1, modules replaced in `site-packages` | 0.25 stock, nothing patched |
| Python / torch | 3.10 / 2.5 | 3.11 / 2.11 |
| Linear solver | per-solve factorization | cuDSS, symbolic phase cached per sparsity pattern, values updated in place |
| Precision | double | double and single; single is the useful default |
| Sensitivities | — | never assembled; `Jtvec` is one reverse pass, and a batched adjoint forms the whole matrix in 0.31 s where it is wanted (30.0 s for SimPEG's stored one on the same problem) |
| Optimizers | — | `TorchLBFGS` and `TorchGaussNewton` with an Occam beta search, both driven by SimPEG's own directives |
| Wavenumbers | SimPEG's, fitted over the mesh | fitted over the survey's electrode distances with positive weights |
| Beyond DC | — | induced polarization in 2.5-D and 3-D, joint non-linear DC-IP, complex resistivity |
| Verification | — | forward parity 7 × 10⁻¹⁶ to 2 × 10⁻¹⁵ against stock SimPEG and gradients to 10⁻⁷ against finite differences, over {2.5-D, 3-D} × {tensor, octree} × {GPU, CPU}, in CI |

On SimPEG's published three-dimensional tutorial the same inversion that takes
692.5 s conventionally takes 78.5 s in double precision and 35.4 s in single,
in the same number of iterations and to the same model. Three-dimensional
tensor meshes work, which they did not before.

## Citing

`CITATION.cff` carries the software citation. The engine and its benchmarks are
described in a manuscript in preparation. Cite Rincon et al. (2026) for the
predecessor and for automatic-differentiation ERT itself; see *Relation to
ADERT* above for where to find it.

## License

MIT. See `LICENSE`.
