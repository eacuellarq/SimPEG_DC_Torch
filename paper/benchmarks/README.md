# Benchmarks

The cases here are the ones whose data anyone can obtain:

- **SimPEG's published three-dimensional tutorial** — the two-sphere model, the
  benchmark Section 3.2 of the manuscript reports.
- **The Century line**, in `data_century/`, distributed with the Transform 2020
  SimPEG tutorial under a permissive licence.
- **A field-scale synthetic**, generated analytically from fixed seeds by the
  `synth3d_*` scripts.
- **AD-TLERT's own synthetics**, from its MIT-licensed repository.

The manuscript was also developed against two field campaigns whose data are
proprietary. Those are not part of this release, and neither are the scripts
that read them; everything the manuscript reports can be reproduced from the
public cases above.

## Where results go

Set `DCTORCH_BENCH_OUT` to a directory that exists. Without it the scripts look
for a `_gpubench` directory beside the repository root, which is where they were
run; create it, or point the variable elsewhere.

## What you need

The engine (`pip install -e .` from the repository root) and a CUDA card for the
GPU paths. The CPU paths run without one, more slowly: every test in
`dctorch/tests` passes on both, and the workflow in `.github/` runs the CPU half
on every push.
