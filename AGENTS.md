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
| `Algorithms/gpu_lisa.cpp`, `Algorithms/lisa_kernel.cl`, `Algorithms/localjc_kernel.cl` | OpenCL path fixed to reproduce the CPU test bit for bit (may become a separate PR: ask the user) |
| `Algorithms/GNUmakefile` | compiles `metal_lisa.mm` on macOS |
| `GeoDamake.macosx.opt` | `.mm` rules, `-framework Metal -framework Foundation` |
| `Explore/LisaCoordinator.cpp` | try Metal before OpenCL on macOS |
| `Explore/MLJCCoordinator.cpp` | try Metal before OpenCL on macOS |
| `.github/workflows/metal_ci.yml` | builds and runs the test on macOS runners |

### Partial copy

- `BuildTools/macosx/GNUmakefile`: take **only** the two lines that copy
  `lisa_kernel.metal` and `localjc_kernel.metal` to `build/GeoDa.app/Contents/Resources`.
  Everything else changed in that file (ad-hoc code signing, extra `.cl` copies) is fork-only.

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
- Kernels draw exactly the CPU's random indices (`round(ThomasWangHashDouble(key) * (n-1))`
  in integer arithmetic). The random sequence of observation `i` starts at `seed + i`,
  as in the OpenCL kernels. With the same sequence the test requires **identical**
  p-values, not "close" ones.
- Apple Silicon only (`MTLGPUFamilyApple7`, i.e. M1 and later); Intel Macs keep using OpenCL.
- No fp64 on Apple GPUs: doubles are passed as (high, low) float pairs, sums use TwoSum,
  kernels are compiled with fast math **off**. Permutations tied with the observed value
  (difference below `tie_tol`) always count as larger, as the CPU's `>=` in
  `LisaCoordinator::ComputeLarger()` intends (note: the OpenCL kernel uses `>`); the CPU
  decides such ties by roundoff, which the test accounts for. Row-standardized weights are assumed.
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
