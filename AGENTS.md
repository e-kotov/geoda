# AGENTS.md — `feature/metal-gpu-acceleration` (development branch, NOT the PR branch)

This branch of the `e-kotov/geoda` fork adds Apple Metal GPU acceleration of the
conditional permutation tests (Local Moran, Local Join Count). It is a **messy
development branch**: besides the feature it carries fork-only tooling (ad-hoc code
signing, fork release workflow, benchmark tool, this file). It will never be merged
as is. The upstream PR is prepared on a separate clean branch, see below.

## Preparing the clean PR branch

1. `git fetch upstream && git switch -c metal-gpu upstream/master`
2. Copy **only** the files of the PR list from this branch:
   `git checkout feature/metal-gpu-acceleration -- <files>`
3. Apply the partial change to `BuildTools/macosx/GNUmakefile` by hand (see list).
4. Double check: `git diff --stat upstream/master` must show exactly the PR list,
   nothing from the "never copy" list, and no `AGENTS.md`.
5. Build and run the test (below); it must end with `ALL METAL TESTS PASSED`.
6. If you add, rename or drop a file on this branch, update the lists below in the
   same commit. Before preparing the PR, re-verify the lists against
   `git diff --stat upstream/master...feature/metal-gpu-acceleration`: every changed
   file must be in exactly one of the lists.

### PR list (copy whole file)

| File | Purpose |
|:---|:---|
| `Algorithms/metal_lisa.h` | API: `is_metal_supported()`, `metal_lisa()`, `metal_localjoincount()` |
| `Algorithms/metal_lisa.mm` | Metal host code (needs `-fobjc-arc`) |
| `Algorithms/lisa_kernel.metal` | Local Moran permutation kernel |
| `Algorithms/localjc_kernel.metal` | Local Join Count permutation kernel |
| `Algorithms/test_metal_lisa.mm` | standalone GPU vs CPU test |
| `Algorithms/test_data_guerry.h` | test data (generated) |
| `Algorithms/test_data_natregimes.h` | test data (generated) |
| `Algorithms/make_test_data.R` | generator of the test data headers |
| `Algorithms/gpu_lisa.cpp`, `Algorithms/lisa_kernel.cl`, `Algorithms/localjc_kernel.cl` | OpenCL path fixed to reproduce the CPU test bit for bit, keys re-designed (GPU-7, needs its own argument: `dev-notes/UPSTREAM_BUGS.md`, `dev-notes/repro/gpu7_shared_noise/`) (may become a separate PR: ask the user) |
| `Algorithms/GNUmakefile` | compiles `metal_lisa.mm` on macOS |
| `GeoDamake.macosx.opt` | `.mm` rules, `-framework Metal -framework Foundation` |
| `Explore/LisaCoordinator.cpp` | try Metal before OpenCL on macOS |
| `Explore/MLJCCoordinator.cpp` | try Metal before OpenCL on macOS |
| `.github/workflows/metal_ci.yml` | builds and runs the test on macOS runners |

### Partial copy

- `BuildTools/macosx/GNUmakefile`: take **only** the two lines that copy
  `lisa_kernel.metal` and `localjc_kernel.metal` to `build/GeoDa.app/Contents/Resources`.
  The two lines copying `lisa_kernel.cl` and `localjc_kernel.cl` to `Contents/Resources` fix upstream bug GPU-5
  (`dev-notes/UPSTREAM_BUGS.md`: the app reads kernels from `Resources`, upstream installs them only to `Contents/MacOS`):
  they belong to the OpenCL-fix PR. Everything else changed in that file (ad-hoc code signing) is fork-only.

### Never copy (fork-only)

`AGENTS.md`, `dev-notes/` (bug register `dev-notes/UPSTREAM_BUGS.md`: source material for SEPARATE upstream PRs and issues), `Algorithms/benchmark_lisa.mm`, `.github/workflows/benchmark.yml`,
`.github/workflows/fork_release.yml`, `.github/workflows/osx_build.yml`,
`BuildTools/macosx/create-dmg/geoda.sh`, `BuildTools/macosx/install_name.py`.
`GdaConst.cpp` must be identical to upstream (GPU stays off by default).

## Test

```bash
clang++ -std=c++14 -O2 -Wall -fobjc-arc Algorithms/test_metal_lisa.mm \
    -framework Metal -framework Foundation -o /tmp/test_metal_lisa
/tmp/test_metal_lisa Algorithms/lisa_kernel.metal Algorithms/localjc_kernel.metal
```

Benchmark (fork-only): same command with `Algorithms/benchmark_lisa.mm` and `-O3`.
Test data comes from GeoDa's own `BuildTools/CommonDistFiles/web_plugins/samples.sqlite`;
regenerate with `Rscript Algorithms/make_test_data.R` (needs R package `sf`).

## Design rules (do not break them)

- The reference is the CPU code: `AbstractCoordinator::CalcPseudoP_range()` and
  `JCCoordinator::CalcPseudoP_range()`. OpenCL (`gpu_lisa.cpp`) can't be a reference:
  its fp64 kernels do not run on Apple Silicon.
- Kernels draw exactly the CPU's random indices (`round(ThomasWangHashDouble(key) * (n-1))` in integer arithmetic).
  Keys (GPU-7): permutation `q` of observation `i` draws from its own key sequence, which starts at
  `TW(TW(seed + i) + q)` (TW = the 64 bit integer hash inside `ThomasWangHashDouble`, 64 bit unsigned arithmetic,
  wraparound intended), in the Metal AND the OpenCL kernels. This is NOT upstream's `seed_start = i + last_seed`
  (one running counter per observation), which makes all observations share their Monte Carlo noise. The key depends
  on (seed, i, q) only, so the result does not depend on the order in which permutations are computed. GPU results
  for a given seed changed with GPU-7: they differ from those of earlier commits of this branch and from upstream's
  OpenCL. With the same keys the test requires **identical** p-values, not "close" ones; `test_permutation_keys()`
  pins the keys with literal values (odd and adjacent `q`, seeds above 32 bits), and every kernel runs against the
  reference with two seeds above 32 bits.
- Apple Silicon only (`MTLGPUFamilyApple7`, i.e. M1 and later); Intel Macs keep using OpenCL.
- No fp64 on Apple GPUs: the host converts each double to a 128 bit fixed point integer and the kernel
  sums exactly (no floating point in the kernels). Float pairs (48 bits) were tried and are WRONG for real
  data: they can neither recognize all exact ties with many neighbors nor resolve genuine differences of
  ~1e-13 (values truncated to 10 decimals). Permutations whose exact difference from the observed value is
  within the CPU's own double rounding band always count as `>=` (the CPU decides those by roundoff).
  Decision 2026-09-18: no soft-float emulation of the CPU's rounding and no exact emulation of the CPU's
  index rounding (both exist, verified, in the untracked `private/softfloat/`, `private/exact_round_final/`):
  the differences are far below seed noise and not worth code or speed. Tests use a BOUNDS rule for tied
  observations with an independently derived CPU error bound.
- Local Join Count: `JCCoordinator` marks isolates undefined: they are never drawn and get no p-value.
  Local Moran instead rejects candidates with `Size() == 0` and keeps self-only observations drawable.
- Kernels return counts; p-values are computed on the host in double, otherwise
  `p <= 0.001` style significance categories break.
- Scope: all OpenCL code that GeoDa actually calls is covered (`gpu_lisa`,
  `gpu_localjoincount`). `gpu_distmatrix` / `distmat_kernel.cl` is dead code upstream
  (never called, kernel not packaged) and is intentionally not ported.
- The GPU code path (Metal and OpenCL) only computes the univariate Local Moran (mean, row-standardized
  weights, one time period, no undefined values) and Local Join Count without undefined values. Upstream
  took it for everything (wrong p-values for e.g. bivariate LISA, verified with GeoDa's OpenCL code); the
  guards in `LisaCoordinator::CalcPseudoP()` and `JCCoordinator::CalcPseudoP()` are part of the PR.
  Still upstream and untouched: the GPU branch ignores `reuse_last_seed == false`.
- The OpenCL kernels were fixed on this branch to give p-values bit-identical to the CPU algorithm
  (verified on PoCL, a CPU OpenCL runtime with fp64; harness in the untracked `private/opencl_fix/`).
- Anti-overfitting rule: besides the fixed datasets, every GPU implementation must pass a held-out
  randomized GPU vs CPU test (untracked `private/fuzz/`) that the implementer does not edit.
- Keep the diff against upstream minimal: no UI, no popups, no timing code in `Explore/`.
