# Upstream bug register: permutation-based local statistics

**Purpose.** This file records defects found in GeoDa desktop (`GeoDaCenter/geoda`) and in
libgeoda (`GeoDaCenter/libgeoda`, the library behind rgeoda and pygeoda) while working on GPU
acceleration of the conditional permutation tests. It exists so that fixes can later be sliced
into separate, self-contained upstream pull requests. It is a register, not a proposal: nothing
here has been reported upstream yet.

**Verification standard.** Every entry was re-derived from the upstream sources
(`git show master:<path>` in the geoda repo, at `f7696a4b`; the libgeoda submodule at `8521405`,
which differs from `upstream/main` only by `sa/UniLOSH.{cpp,h}` and the `gda_losh` entry points
in `gda_sa.{cpp,h}` — everything else cited below is upstream code). A claim is marked CONFIRMED
when a program was run and produced numbers, or when a build/code fact is unambiguous and no
numbers are needed; the two entries where no numbers were produced (LG-4, GPU-5) say so
explicitly. Claims inherited from earlier agent work were re-checked; several turned out to be
wrong or overstated and are kept below with the refutation. Where the effect of a defect is
indistinguishable from Monte Carlo noise, that is said plainly.

**Status vocabulary**

| status | meaning |
|:---|:---|
| CONFIRMED | wrong numbers, reachable by a user today |
| CONFIRMED-LATENT | the code is wrong, but no user-facing path reaches it in the current release |
| COSMETIC-OR-DEAD-CODE | dead, unused or purely stylistic; no numeric effect |
| INTENDED-OR-DISPUTABLE | behaves as written; whether it is "right" is a modelling choice |
| REFUTED | the claim does not hold; see the refutation |

**Reproductions** live in the untracked `private/upstream_bugs/` of this branch
(`desktop_repro/`, `libgeoda_repro/`, `r_repro/`, `multitime/`, `bugC/`, plus `patches/`).
The essential code and numbers are inlined below so that this document stands on its own.

---

## Summary table

| id | title | component | status | reachable by | severity | fixed on this branch |
|:--|:---|:---|:---|:---|:---|:---|
| A | conditional draw uses the last time period's weights | desktop | CONFIRMED | space-time variable with missing values | low normally, high for adversarial masks | no |
| B | per-period isolate flag uses the running maximum neighbour count | desktop | CONFIRMED | same as A | low | no |
| C | GPU branch of Local Moran is handed the unmodified weights | desktop (GPU) | CONFIRMED-LATENT | GPU enabled + kernel file found | medium | yes, `30faf4ff` |
| D | Local Moran: observed is a mean, permuted a sum when `row_standardize == false` | both | CONFIRMED-LATENT | nobody: the flag is dead in both code bases | n/a | no |
| E | Local Geary: every permuted value stays 0 when `row_standardize == false` | desktop | CONFIRMED-LATENT | nobody: checkbox commented out | n/a | no |
| F | Local Geary observed statistic mixes two lag denominators | desktop | CONFIRMED | any kernel weights (they always carry a self-link) | medium | no |
| G | Local Neighbour Match Test permutes only once | desktop | COSMETIC-OR-DEAD-CODE | nobody: the only caller is commented out | n/a | no |
| H | Moran randomization dialog "compares two different statistics" | desktop | REFUTED | — | — | — |
| I | bivariate Local Geary computes nothing | desktop | COSMETIC-OR-DEAD-CODE | nobody: never constructed | n/a | no |
| J | `GalElement::Update()` leaves `nbrLookup` stale | desktop | CONFIRMED-LATENT | no live caller of `GetRW()` on updated weights | n/a | no |
| K | inconsistent fold rules `<` vs `<=` | both | REFUTED | — | — | — |
| L | `using_median` uninitialised in the second `LisaCoordinator` constructor | desktop | CONFIRMED-LATENT | nobody: that constructor is never called | n/a | no |
| M | dead `values` array in the Join Count GPU branch | desktop | COSMETIC-OR-DEAD-CODE | — | n/a | no |
| N | dead locals in two permutation loops | desktop | COSMETIC-OR-DEAD-CODE | — | n/a | no |
| GPU-1 | GPU branch taken for every LISA variant, but only `data1_vecs[0]` is passed | desktop (GPU) | CONFIRMED | GPU enabled + kernel file found | high | yes, `cfceee2b` |
| GPU-2 | the OpenCL Local Moran kernel does not implement the CPU algorithm | desktop (GPU) | CONFIRMED | same | high | yes, `04529b11` |
| GPU-3 | `gpu_localjoincount()` never runs its kernel and returns p = 0 for every observation | desktop (GPU) | CONFIRMED | same | high | yes, `04529b11` |
| GPU-4 | the GPU branch ignores `reuse_last_seed == false` | desktop (GPU) | CONFIRMED | same | medium | no |
| GPU-5 | macOS bundle installs the kernels next to the binary, the code loads them from `Resources` | desktop (GPU) | CONFIRMED | every macOS release user who turns the GPU on | medium (the GPU option cannot work) | no |
| GPU-6 | dead `gpu_distmatrix()`: 6 arguments set on a 4-argument kernel | desktop | COSMETIC-OR-DEAD-CODE | nobody | n/a | no |
| GPU-7 | GPU seeding `seed_start = i + last_seed`: all observations share their Monte Carlo noise | desktop (OpenCL kernels, copied by the Metal port) | CONFIRMED by replay (`dev-notes/repro/gpu7_shared_noise/`) | every GPU run | medium: no bias, but the map is 2.5 to 4.8 times less stable between seeds than the CPU's | planned (re-keying of the four kernels) |
| LG-1 | libgeoda counts a self-link in the permutation size but not in the observed statistic | libgeoda | CONFIRMED | **rgeoda/pygeoda, any kernel weights** | medium (anti-conservative p-values) | no |
| LG-2 | `row_standardize` is dead in libgeoda too; `UniG`/`UniGstar` would degenerate | libgeoda | CONFIRMED-LATENT | nobody | n/a | no |
| LG-3 | NaN observed statistic, and NaN contagion into the cluster map | libgeoda | CONFIRMED | rgeoda with self-only or fully-undefined neighbourhoods | medium | no |
| LG-4 | `permutation_method = "lookup"` draws isolates, `"complete"` does not | libgeoda | CONFIRMED (code only, effect not measured) | `rgeoda::local_*(permutation_method = "lookup")` | unknown, probably low | no |
| LG-5 | LOSH pairs drawn observations with positional weights | libgeoda | REFUTED | — | — | — |
| X | Local Moran / Local Geary discard non-binary weight values in the observed statistic | both | INTENDED-OR-DISPUTABLE | anyone using kernel / inverse-distance weights | documentation-level | no |
| LOSH-1 | **not upstream** — the observed LOSH lag includes the location itself, the null never does | this branch's own libgeoda code | CONFIRMED | rgeoda `local_losh()` with kernel weights | medium | no |

---

## A — the conditional draw uses the last time period's weights

**What happens.** `AbstractCoordinator::CalcPseudoP_range()` draws one permutation and reuses it
for every time period. The pointer used to decide whether a candidate observation may be drawn
is assigned *inside* the per-period loop and read *after* it, so it always holds the weights of
the last period.

**Code.** `Explore/AbstractCoordinator.cpp:515-550` (upstream `master`, identical on this branch):

```cpp
GalElement* w;                                   // :515
int numNeighbors = 0;
for (int t=0; t<num_time_vals; t++) {
    w = Gal_vecs[t]->gal;                        // :520  reassigned every period
    ...
}
...
    if (newRandom != cnt && !workPermutation.Belongs(newRandom)
        && w[newRandom].Size()>0) {              // :550  uses Gal_vecs[num_time_vals-1]
```

The same shape is in `Explore/GStatCoordinator.cpp:674-708` and
`Explore/LocalGearyCoordinator.cpp:849-874`.

`Gal_vecs[t]` differs from `Gal_vecs[t']` only when the undefined mask differs between periods:
`LisaCoordinator::Calc()` (`Explore/LisaCoordinator.cpp:481-487`) makes a copy of the weights and
calls `GalWeight::Update(undefs)` on it, which deletes every link to an undefined observation.

**Reachability.** Desktop, `num_time_vals > 1`, i.e. a time-grouped variable synced with the
global time control (`LisaCoordinator.cpp:263-279`), *and* missing values that differ per period.
Not reachable from rgeoda/pygeoda: libgeoda has no time dimension.

**Reproduction** (`private/upstream_bugs/multitime/`, faithful transcription of
`StandardizeData`/`Calc`/`CalcPseudoP_range`/`ComputeLarger`; 60-observation chain, 2 periods,
999 permutations, seed 123456789):

```
== CLAIM A: draw-rejection weights are the LAST period's ==
  period 0: 59 of 60 p-values differ from the per-period-weights version, max |dp| = 0.0310
            [seed-noise floor: 58 differ, max |dp| = 0.0340]
  period 1: 55 of 60 p-values differ, max |dp| = 0.0460
            [seed-noise floor: 59 differ, max |dp| = 0.0450]
```

**This is the honest result for a realistic mask: the effect is indistinguishable from changing
the seed.** Changing which candidates are eligible reshuffles the whole RNG stream, so *many*
p-values move, but they move by Monte Carlo amounts. The earlier survey did not measure this.

The defect can nevertheless be made large. Second scenario in the same program: period 1 marks
every even-indexed observation undefined, so after `Update()` every *odd* observation is
neighbourless in period 1, and the period-0 draw (which uses period 1's weights) can never draw
an odd observation:

```
== CLAIM A, scenario 2: period 1's mask isolates every odd obs ==
  period 0 p-values: 57 of 60 differ, max |dp| = 0.4930
  significant at p<=0.05 in period 0: upstream 39, per-period weights 20
  upstream period-0 p: 0.001 0.002 0.001 0.004 0.001 0.015 0.001 0.026 0.001 0.067 ...
  correct  period-0 p: 0.494 0.262 0.262 0.245 0.265 0.224 0.260 0.232 0.244 0.230 ...
```

So the severity is data dependent: normally noise, but a period whose missing-value pattern
isolates a systematically selected subset makes period 0's null distribution draw from a biased
pool.

**Proposed fix.** `private/upstream_bugs/patches/bug-A-draw-weights-per-period.patch` introduces
`GalElement* w_draw = Gal_vecs[0]->gal;` before the loop and uses it in the rejection test, in
all three coordinators. **This is a judgement call and needs a maintainer decision**: one draw
serves all periods, so no single period's weights are "correct". The conservative choice above
leaves the single-period case (the overwhelmingly common one) bit-identical. The principled
alternative is to reject on the *unmodified* weights (`Gal_vecs_orig[t]`, period-independent by
construction), but that would also change single-period results whenever there are missing
values, which is a bigger behavioural change than a bug fix should carry.

**Testing a fix.** Run the harness above; the two scenarios must give per-period-stable results,
and the single-period no-missing-value case must be bit-identical to the current code.

---

## B — the per-period isolate flag uses the running maximum neighbour count

**The survey's claim is refuted; a different defect at the same place is confirmed.**

*Refutation of the original claim.* The survey asserted that
`if (w[cnt].Size() > numNeighbors) { numNeighbors = Size(); if (self) numNeighbors -= 1; }`
does not compute the maximum over periods. It does. Write `S(t)` for `Size()` and `r(t)` for the
self-excluded count, `c` for `numNeighbors` before period `t`. If `S(t) > c` then
`c := r(t) >= S(t)-1 >= c`, so the new value is `max(c, r(t))`. If `S(t) <= c` then
`r(t) <= S(t) <= c`, so leaving `c` alone is also `max(c, r(t))`. By induction the loop yields
exactly `max_t r(t)`. Both worked examples in the survey evaluate to the correct answer.

*What is actually wrong.* `Explore/AbstractCoordinator.cpp:528-531`:

```cpp
    int* _sigCat = sig_cat_vecs[t];
    if (numNeighbors == 0) {          // <- the RUNNING MAXIMUM, not this period's count
        _sigCat[cnt] = 6;             // 6 == "isolate" in the significance map
    }
```

An observation that has neighbours in period 0 but loses all of them in period 1 (because they
are all undefined in period 1) is *not* flagged as an isolate in period 1, and it still gets a
permutation of `max_t r(t)` draws. Its observed statistic for that period is 0 (`Calc()` takes
the `W[i].Size() == 0` branch at `LisaCoordinator.cpp:500-504` and leaves `localMoran[i] = 0`),
so it is compared against a permuted value of essentially arbitrary sign.

**Reachability.** Same as A. The order of the `t` loop matters: the reverse case (isolate in
period 0, neighbours in period 1) is handled correctly.

**Reproduction** (same program): 60-observation chain, obs 11 is *defined* but its two
neighbours are undefined in period 0, obs 41 likewise in period 1:

```
  period 0 obs 11: upstream sig_cat=0 p=0.4870   per-period-correct sig_cat=6
  period 1 obs 41: upstream sig_cat=0 p=0.4760   per-period-correct sig_cat=6
```

The p-value is not extreme — it is meaningless. The user-visible symptom is that the
significance map shows the location as "not significant" while the cluster map shows it as
"neighbourless", i.e. the two maps disagree.

**Proposed fix.** `private/upstream_bugs/patches/bug-B-per-period-isolate.patch`: compute the
period's own `numNeighbors_t` (self excluded) and use it both for the maximum and for the
isolate flag. Behaviour is unchanged for `num_time_vals == 1`.

**Testing a fix.** The harness above; the two listed observations must come out as `sig_cat = 6`,
everything else unchanged.

---

## C — the GPU branch of Local Moran is handed the unmodified weights

**What happens.** `LisaCoordinator::CalcPseudoP()` passes `weights->gal` to `gpu_lisa()`, but the
observed statistic it passes alongside (`local_moran_vecs[0]`) was computed from
`Gal_vecs[0]->gal`, a copy with links to isolates and undefined observations removed.

**Code.** `Explore/LisaCoordinator.cpp:566` on upstream `master`:

```cpp
        double* values      = data1_vecs[0];
        double* local_moran = local_moran_vecs[0];
        GalElement* w       = weights->gal;        // :566  <-- original, not Gal_vecs[0]
```

and `:579` in the same function recomputes `int numNeighbors = w[cnt].Size();` without the
self-neighbour correction that `AbstractCoordinator.cpp:523-526` applies, so an observation whose
only neighbour is itself is not marked as an isolate on the GPU path.

**Reachability.** Only with `GdaConst::gda_use_gpu == true` (Preferences → System → "use GPU",
default off, `GdaConst.cpp:332`) *and* an OpenCL device with `cl_khr_fp64` *and* the kernel file
actually being found — see GPU-5, which means this is not reachable on a macOS release at all.
Not applicable to libgeoda (no GPU path).

**Reproduction** (`private/upstream_bugs/bugC/`). 50×50 rook lattice, 26 observations turned into
isolates by clearing their own neighbour lists only (asymmetric weights, as produced by a `.gal`
file with empty rows or a knn matrix read as GAL); the CPU permutation is run twice against the
*same* observed statistic, once with each weight set:

```
50x50 rook lattice, 26 asymmetric isolates, 101 observations lost a link
observations tested: 2474
p-values differing between Gal_vecs[0] (CPU) and weights->gal (GPU): 100
max |dp| = 0.0510
significant at p<=0.05: CPU 1109, GPU weights 1125, flips 16
```

With symmetric contiguity weights and no missing values the two weight sets are identical and
nothing changes. With missing values the effect is larger, but there it is dominated by GPU-1
(the GPU path ignored `undefs` entirely).

**Fixed on this branch**: `30faf4ff` ("fix: GPU branch of Local Moran uses the same weights as
the CPU"), one line, `weights->gal` → `Gal_vecs[0]->gal`.

**Testing a fix.** The reproduction above, plus the branch's `Algorithms/test_metal_lisa.mm`
lattice case, which contains isolates and a self-neighbour.

---

## D — Local Moran: the observed value is a mean, the permuted values are sums, when `row_standardize == false`

**What happens.** The observed statistic always divides by the neighbour count; the permuted one
divides only when `row_standardize` is true. The two are then not on the same scale and the
pseudo p-value is meaningless.

**Code (desktop).** `Explore/LisaCoordinator.cpp:521-534` computes
`Wdata = W[i].SpatialLag(data1, is_binary=true, i)`, and that overload
(`ShapeOperations/GalWeight.cpp:239-262`) ends with `if (n_nbrs > 0) lag /= (double) n_nbrs;`
at `:257` — unconditionally. `row_standardize` appears exactly once in the whole file, in
`ComputeLarger()` at `:660`:

```cpp
            if (validNeighbors > 0 && row_standardize) {
                permutedLag /= validNeighbors;
            }
            const double localMoranPermuted = permutedLag * data1[cnt];
```

**Code (libgeoda).** The same mismatch: `sa/UniLocalMoran.cpp:71` `sp_lag = sp_lag / nn;` is
unconditional, while `sa/UniLocalMoran.cpp:101` and `:124` divide only
`if (validNeighbors > 0 && row_standardize)`. `sa/UniGeary.cpp:107,138` and
`sa/BiLocalMoran.cpp:104,127` have the same shape. (The survey stated that libgeoda's `UniGeary`
"has no such guard and is fine" — that is wrong; it has the guard, it simply assigns the result
outside it, which gives the D pattern rather than the E pattern.)

**Reachability — the survey's claim is refuted.**
*Desktop*: `LisaWhat2OpenDlg::m_RowStand` is set to `true` in the constructor and the checkbox is
commented out (`DialogTools/LisaWhat2OpenDlg.cpp:34` and `:63`), so only `true` ever reaches the
coordinator. *libgeoda/rgeoda*: `row_standardize` is hard-coded `true` in **both** `LISA`
constructors (`sa/LISA.cpp:79`, `:107`); the only way to change it is `LISA::SetRowStandardize()`
(`sa/LISA.cpp:711`), and a grep over the whole of `rgeoda/src` (excluding the vendored geoda
tree) finds **no call site at all**. None of `local_moran`, `local_bimoran`, `local_geary`,
`local_multigeary`, `local_g`, `local_gstar`, `local_joincount`, `local_losh` exposes such an
argument (verified programmatically over the installed package's namespace). The survey's
statement that this is "reachable through libgeoda/rgeoda (`row_standardize` parameter of
`local_moran`)" is **wrong**: there is no such parameter. The bug is CONFIRMED-LATENT in both
code bases.

**Reproduction.** Desktop logic lifted into `private/upstream_bugs/desktop_repro/`
(999 permutations, seed 123456789, single-threaded seeding). The observed `localMoran` array is
**bit-identical** (`memcmp == 0`) for the two settings, so only the null moves:

| data | row_standardize | p≤0.05 | p≤0.01 | min p | mean p |
|---|---|---|---|---|---|
| Guerry, n=85 | TRUE | 27 | 9 | 0.0010 | 0.1830 |
| Guerry, n=85 | FALSE | **0** | 0 | 0.1300 | 0.4057 |
| natregimes, n=3085 | TRUE | 1168 | 513 | 0.0010 | 0.1473 |
| natregimes, n=3085 | FALSE | **1** | 0 | 0.0170 | 0.4094 |

84/85 and 3058/3085 p-values differ. **Correction to the earlier framing:** p does *not*
collapse to a constant or to 1. `AbstractCoordinator.cpp:568` folds with
`if (permutations-countLarger <= countLarger)`, which caps LISA's p at 0.5005, and the permuted
sum has standard deviation ≈ `sqrt(k)` while the observed threshold is of order 1, so the count
sits near `permutations/2` and p drifts to ≈ 0.41. **The failure mode is a total loss of power,
not extreme p-values.** libgeoda behaves the same way: with `SetRowStandardize(false)` forced in
a harness, `UniLocalMoran` on Guerry goes from 27 significant to 0 and the minimum p rises from
0.001 to 0.130.

**Proposed fix.** Three options, in increasing scope:
1. `private/upstream_bugs/patches/bug-D-lisa-rowstandardize.patch` — drop `&& row_standardize`
   from `ComputeLarger()`, so the permuted lag is normalised the same way as the observed one.
   One line, no behaviour change for any reachable input.
2. Implement binary weights properly: make the *observed* statistic a sum when
   `row_standardize == false` (the textbook `I_i = z_i Σ_j w_ij z_j`). This changes the reported
   Local Moran values and the Moran scatter plot, so it is a feature, not a fix.
3. Remove the dead parameter from `LisaCoordinator`/`AbstractCoordinator` and from `LISA` in
   libgeoda.
Option 1 plus a note is the smallest honest change; option 3 is what a maintainer may prefer.

**Testing a fix.** The harness above must show identical p-values for `row_standardize` true and
false once the permuted lag is normalised (option 1), and Guerry/natregimes with
`row_standardize == true` must be bit-identical to today.

---

## E — Local Geary: every permuted value stays 0 when `row_standardize == false`

**Code.** `Explore/LocalGearyCoordinator.cpp:846` allocates
`gci[t].resize(permutations, 0);`, and the only write to `gci[t][perm]` is inside a guard —
`:977` for the univariate branch, `:941` for the multivariate one:

```cpp
                    //NOTE: we shouldn't have to row-standardize or multiply by data1[cnt]
                    if (validNeighbors && row_standardize) {
                        if (_data1_square && _data1) {
                            gci[t][perm] = _data1_square[cnt] - 2.0*_data1[cnt]*wwx/validNeighbors
                                           + wwx2/validNeighbors;
                        }
                    }
```

With binary weights every permuted Local Geary is exactly 0, and the p-value at `:1032` depends
only on the sign of the observed value. Local Geary has **no** "pick the smallest" fold
(contrast `AbstractCoordinator.cpp:568`), so `p = 1.0` is genuinely emitted.

**Reachability.** `LocalGearyWhat2OpenDlg::m_RowStand` is hard-wired `true` with the checkbox
commented out (`DialogTools/LisaWhat2OpenDlg.cpp:129` and `:156`). CONFIRMED-LATENT.
libgeoda's `UniGeary` has the D pattern instead (see §D) and is equally unreachable.

**Reproduction** (`private/upstream_bugs/desktop_repro/`):

| data | row_standardize | p≤0.05 | p = 1/1000 | p = 1.0 | mean p |
|---|---|---|---|---|---|
| Guerry | TRUE | 26 | 1 | 0 | 0.1886 |
| Guerry | FALSE | **85 / 85** | **85** | 0 | 0.0010 |
| natregimes | TRUE | 1252 | 191 | 0 | 0.1588 |
| natregimes | FALSE | **3074** | **3074** | **11** | 0.0046 |

The result is deterministic and independent of the RNG: `localGeary != 0` → p = 1/1000 (the most
significant value the tool can produce), `localGeary == 0` → p = 1.0. Since
`localGeary = mean_j[(z_i - z_j)^2] >= 0`, essentially the whole map would be drawn as
significant. In natregimes, 16 observations sit at or just below 0 (neighbourhoods of identical
`HR90` values) and their p flips between 0.0010 and 1.0000 on a −1.11e-16 rounding difference.

**Proposed fix.** `private/upstream_bugs/patches/bug-E-localgeary-rowstandardize.patch`: assign
`gci[t][perm]` whenever `validNeighbors > 0`, keeping the division (which matches the observed
statistic, as in §D option 1). Both the univariate and the multivariate branch.

**Testing a fix.** As for D: the two settings must give identical p-values after the fix, and
`row_standardize == true` must be unchanged.

---

## F — Local Geary's observed statistic mixes two different lag denominators

**What happens.** The two spatial lags that make up the observed Local Geary are computed with
different overloads and therefore different denominators, while the permuted value uses one
denominator for both.

**Code.** `Explore/LocalGearyCoordinator.cpp:724-726`:

```cpp
                Wdata  = W[i].SpatialLag(data1, is_binary, i);   // divides by the NON-SELF count
                Wdata2 = W[i].SpatialLag(data1_square);          // default self_id = -1
            ...
            localGeary[i] = data1_square[i] - 2.0 * data1[i] * Wdata + Wdata2;
```

`ShapeOperations/GalWeight.h:45` declares `double SpatialLag(const double *x, bool is_binary = true,
int self_id = -1) const;`, and the `self_id < 0` branch (`GalWeight.cpp:245-247`) divides by
`Size()`, which includes a self-link, and does not divide at all when `Size() == 1`.
The permutation (`:977`) divides both terms by `validNeighbors`.

**Reachability — broader than the survey assumed.** Self-links are present in *every* GeoDa
kernel weights matrix, not only when "apply kernel to the diagonal" is ticked:
`SpatialIndAlgs::knn_build()` pushes the self entry unconditionally when a kernel is in use
(`SpatialIndAlgs.cpp:294-300`, comment `// add self if kernel weights`; the 3-d twin at `:398`), and
`SpatialIndAlgs::apply_kernel()` at `:233` only *rewrites the weight value* of that entry when
the checkbox is off (`if (!use_kernel_diagnals && i == nbrs[j].nbx) { nbrs[j].weight = 1.0;
continue; }`).
`.kwt` files are accepted by `WeightsNewManager::GetGal()` (`ShapeOperations/WeightsManager.cpp:377-397`)
through `ReadGwtAsGal`, and `GalElement::RemoveSelfNeighbor()` (`GalWeight.cpp:120-134`), which
was written for exactly this situation, is **never called anywhere in the tree**. So any Local
Geary run on kernel weights hits this.

**Reproduction** (`private/upstream_bugs/desktop_repro/`, Guerry, 999 permutations):

| weights | observed ≠ mean_j[(z_i−z_j)²] beyond roundoff | max abs err | mean abs err | p differing from a corrected version |
|---|---|---|---|---|
| queen, no self-link | **0 / 85** | 5.3e-15 | 2.2e-16 | **0 / 85** |
| queen + self-link | **85 / 85** | **4.5906** | **0.2100** | **76 / 85** |

With self-links, `p ≤ 0.01` goes from 11 to 14 and one observation gets a *negative* observed
Local Geary. Worked example, `i = 0`, k = 4 real neighbours: `Wdata2` upstream 0.46483354 vs
0.55311878 corrected, `localGeary` 0.09165960 vs 0.17994484, p 0.0690 vs 0.1960. The bias has a
closed form: `(k·z_i² − Σ_{j≠i} z_j²) / (k(k+1))`.

The `Size() == 1` case in `if (sz>1) lag /= sz` turns out to make **no** difference (dividing by
1 is a no-op) — that part of the survey's claim is refuted. The genuine degenerate case is an
observation whose only neighbour is its self-link: `Wdata = 0`, `Wdata2 = z_i²`, so
`localGeary = 2 z_i²` (4.5 instead of 0 in the test case), and no p-value is computed for it at
all because `numNeighbors = Size() - 1 = 0` makes `CalcPseudoP_range()` `continue` at `:862`.

**Proposed fix.** Pass the self id to the second lag as well:
`Wdata2 = W[i].SpatialLag(data1_square, is_binary, i);`. One argument. It changes reported Local
Geary values for kernel weights, which is the point — the current ones are not
`mean_j[(z_i − z_j)²]`. An alternative, arguably better, fix is to call
`GalElement::RemoveSelfNeighbor()` once when weights are loaded for a LISA-type computation,
which would also fix the `Size()==1`-self case and LG-1's desktop analogue in one place. No patch
file is provided because the choice between the two is a maintainer decision.

**Testing a fix.** The harness above: with a self-link added to every observation, the observed
value must equal `mean_j[(z_i − z_j)²]` to roundoff for all 85 observations; without self-links
nothing may change.

---

## G — the Local Neighbour Match Test permutes only once (dead code)

**Code.** `DialogTools/nbrMatchDlg.cpp:994-1042`:

```cpp
    int rand=0, newRandom, countLarger=0;      // :999   declared OUTSIDE the loop
    ...
    for (size_t i=0; i<permutations; ++i) {
        while (rand < nbr_sz) { ... }          // :1002  only true for i == 0
        for (int cp=0; cp<nbr_sz; cp++) {
            perm_nbr = workPermutation.Pop();  // :1019  returns -1 once the set is empty
            if (variable_w->CheckNeighbor(idx, perm_nbr)) match += 1;
        }
```

`GeoDaSet::Pop()` returns `-1` on an empty set (`GenUtils.h:568`) and
`GalWeight::CheckNeighbor(idx, -1)` is `false` (`GalWeight.cpp:410` → `:47`), so permutations
1..P−1 all produce `match = 0`.

**Reachability — the survey's "it is broken today" is refuted.** The only construction of
`LocalMatchCoordinator` in the whole tree is inside a `/* ... */` block at
`DialogTools/nbrMatchDlg.cpp:477-484`, with the comment
`// run perm sig, ignore for now since p = C(k,v).C(N-k,k-v) / C(N,k) is used`. The shipped tool
uses a closed-form hypergeometric probability instead. **COSMETIC-OR-DEAD-CODE.**
libgeoda's `gda_neighbor_match_test` is a different implementation and does not contain this
pattern.

**Reproduction** (`private/upstream_bugs/desktop_repro/`, 60-point synthetic, cardinalities 1–4,
999 permutations), showing what would happen if the code were re-enabled:

| variant | p distribution | p ≤ 0.05 |
|---|---|---|
| upstream | **p = 0.0010 for 55 obs, p = 0.0020 for 5 obs, nothing else** | **60 / 60** |
| `int rand = 0` moved inside the loop | 0.0010 (3), 0.0030–0.0160 (22), 0.0800–0.1260 (34), 0.4830 (1) | **25 / 60** |

Every location would be reported as significant at p ≤ 0.01.

**Proposed fix.** `private/upstream_bugs/patches/bug-G-nbrmatch-reset-rand.patch` moves the
declaration inside the permutation loop. Worth doing even for dead code, because the natural
next step for a maintainer is to re-enable the block.

**Testing a fix.** The harness above; also assert that `workPermutation.Size() == 0` at the end
of each permutation.

---

## H — REFUTED: the Moran randomization dialog does not compare two different statistics

**The claim.** The observed global Moran's I shown in the randomization histogram is a regression
slope (`DialogTools/RandomizationDlg.cpp:326-330`) while the permuted ones are
`Σ lag·x / (n_valid − 1)` (`:390-408`), so the reference line and the null distribution would be
incomparable.

**Why it is wrong.** The data handed to the dialog are the *already z-standardized*
`lisa_coord->data1_vecs[xt]` (`Explore/LisaScatterPlotView.cpp:872-880`), standardized with the
sample standard deviation (`GenUtils::StandardizeData`, `GenUtils.cpp:1796-1820`, divisor
`nValid − 1`). `SimpleLinearRegression` (`GenUtils.cpp:1015-1033`) computes
`beta = covariance / varX` with `covariance = E[XY] − x̄ȳ` and `varX = Σ(x−x̄)²/n`, i.e.

```
beta = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²  =  Σ x·y / (nValid − 1)      (because x̄ = 0 and Σx² = nValid − 1)
```

which is exactly the permuted expression. The two are the same statistic. The commented-out
original code just above `CalcMoran()` computes the same thing directly.

**A smaller, real defect at the same place.** Inside the permutation loop
(`RandomizationDlg.cpp:383-408`) the skip test is `if (undefs[perm[i]] || W[i].Size() == 0)
continue;` — it drops a term when the *permuted* value is undefined, so each permutation sums a
random number of terms, while the divisor `valid_num_obs - 1` is constant and the observed value
sums over all valid `i`. With missing values present the null is therefore slightly deflated.
Also, the exact equality above only holds when the regression's valid set equals the
standardization set; an observation that is defined but lost all its neighbours to `Update()` is
in the second set and not in the first. Both are small and only bite with missing data. Severity:
low. No patch provided.

---

## I — bivariate Local Geary computes nothing (dead code)

**Code.** `Explore/LocalGearyCoordinator.cpp:955-962` accumulates `permutedLag` from `_data2` and
then never uses it: `gci[t][perm]` is built from `wwx`/`wwx2`, which stay 0 in the bivariate
branch, so every permuted value equals `_data1_square[cnt]`. The observed value at `:722-723`
sets `Wdata2 = 0`.

**Reachability.** `LocalGearyCoordinator::bivariate` exists in the enum
(`Explore/LocalGearyCoordinator.h:70`) but the only two constructions in the tree are
`GeoDa.cpp:3878` (`univariate`) and `GeoDa.cpp:3931` (`multivariate`).
**COSMETIC-OR-DEAD-CODE.** libgeoda has no bivariate Geary at all.

**Proposed fix.** Delete the branch, or implement it. No patch provided.

---

## J — `GalElement::Update()` leaves `nbrLookup` stale

**Code.** `ShapeOperations/GalWeight.cpp:137-162` erases entries from `nbr` and `nbrWeight` and
erases the removed key from `nbrLookup`, but does not renumber the positions stored for the
*surviving* neighbours — unlike `GalElement::RemoveSelfNeighbor()` at `:121-134`, which rebuilds
the map. `nbrAvgW` and `is_nbrAvgW_empty` are not invalidated either.

**Reproduction** (`private/upstream_bugs/desktop_repro/`), `nbr = [10,20,30,40]`,
`nbrWeight = [1,2,3,4]`, `Update()` removing id 20:

```
nbrLookup = {10->0, 30->2, 40->3}      <- surviving positions not renumbered
Check(10)=1 Check(20)=0 Check(30)=1 Check(40)=1     <- key membership still correct
GetRW(10) = 0.125   (correct 1/8)
GetRW(30) = 0.500   (correct 3/8)      <- returns 4/8, neighbour 40's weight
GetRW(40) : nbrLookup[40]==3 but nbrAvgW.size()==3  -> out-of-bounds read (UB)
```

`is_nbrAvgW_empty` is also not reset, so a `GetRW()` call made *before* `Update()` leaves a
stale cache behind (0.400 before and after, correct value 0.500).

**Reachability — no live path.** `GetRW()` is called only from `Regression/smile2.cpp:147,154`
(inside `T()`) and `:161,162` (inside a comment). `T()` is called once from `smile2.cpp:747` with
an array built at `DialogTools/RegressionDlg.cpp:480-515`, which is either `gw->gal` straight
from the weights manager or a freshly `SetNbr()`-built array — never an `Update()`d copy. Every
call site of `Update()` (`LisaCoordinator.cpp:484`, `LocalGearyCoordinator.cpp:599,703`,
`GStatCoordinator.cpp:295`, `MLJCCoordinator.cpp:338`, `RandomizationDlg.cpp:674,689,697,744,756,764`,
`ShapeOperations/RateSmoothing.cpp:227,329`) operates on a fresh `new GalWeight(*W)` copy and at
most once. **CONFIRMED-LATENT.** Note the "at most once" matters: `Update()` itself reads
`nbrLookup[obj_id]` positionally at `:143`, so a second call on the same element would delete the
wrong neighbours.

**Proposed fix.** `private/upstream_bugs/patches/bug-J-galelement-update-lookup.patch` rebuilds
`nbrLookup` and clears the `nbrAvgW` cache at the end of `Update()`, mirroring
`RemoveSelfNeighbor()`. Cheap insurance that also makes `Update()` idempotent-safe.

**Testing a fix.** The four-neighbour unit case above, plus a check that `Update()` applied twice
gives the same result as applying it once with the union of the masks.

---

## K — REFUTED: the two fold rules are algebraically identical

**The claim.** `AbstractCoordinator.cpp:568` folds with
`if (permutations - countLarger <= countLarger)`, while `GStatCoordinator.cpp:761,766`,
`MLJCCoordinator.cpp:654` and libgeoda's `UniG`/`UniGstar` use `<`; the survey said this gives
different answers when `countLarger == permutations/2`.

**Why it is wrong.** The two rules differ only when `permutations − countLarger == countLarger`,
and in exactly that case the assignment `countLarger = permutations − countLarger` writes back
the value it already had. The outputs are identical for every input, for any number of
permutations, odd or even. This is a readability wart, not a defect.

The related observation that `LocalGearyCoordinator` does not fold at all is a *different*
p-value definition (the direction is chosen by comparing the observed value with the mean of the
permuted ones, `LocalGearyCoordinator.cpp:993-1032`), which is the standard construction for
Local Geary. INTENDED.

---

## L — `using_median` is never initialised in the second `LisaCoordinator` constructor

**Code.** `Explore/LisaCoordinator.h:75` declares `bool using_median;` with no initialiser. The
first constructor initialises it in its member list (`LisaCoordinator.cpp:82`); the second one
(`LisaCoordinator.cpp:94-185`, `LisaCoordinator(wxString weights_path, ...)`) never assigns it,
and it is read at `:491` (`using_median ? GenUtils::Median(...) : 0`) and in `ComputeLarger()`.

**Reachability.** That constructor has no caller: all nine constructions in the tree
(`GeoDa.cpp:3551, 3623, 3682, 3736, 3981, 4037, 4090, 4167, 4226`) use the
`boost::uuids::uuid weights_id` overload. **CONFIRMED-LATENT** — but it is a live trap for the
next person who uses that constructor, and it is also read by the GPU guard added on this branch.
`LocalGearyCoordinator` has an analogous unused second constructor (`:120`).

**Proposed fix.** `private/upstream_bugs/patches/bug-L-using-median-init.patch` gives the member a
default initialiser in the header. One line.

---

## M, N — dead code in the permutation paths

* **M**: `Explore/MLJCCoordinator.cpp:476-484` builds a `double* values = new double[num_vars*num_obs]`,
  fills it, and `delete[]`s it at `:496` without ever passing it anywhere.
  `gpu_localjoincount()` takes `zz`, not `values`.
* **N**: `Explore/LocalGearyCoordinator.cpp:921` declares `double permutedLag = 0;` that is only
  ever written in the dead bivariate branch (§I) and never read. The survey also listed
  `Explore/LisaCoordinator.cpp:509-513` as dead locals — **that part is refuted**: those lines are
  the median branch's `int nn = W[i].Size(); if (W[i].Check(i)) nn -= 1;`, and `nn` is used
  immediately to size `nbr_data`.

COSMETIC-OR-DEAD-CODE, no numeric effect, no patch. Worth folding into whichever PR touches that
file.

---

## GPU-1 — the GPU branch is taken for every LISA variant, but only `data1_vecs[0]` is passed

**Code.** `Explore/LisaCoordinator.cpp:556-601` on upstream `master` has no guard:

```cpp
    if (GdaConst::gda_use_gpu == false) {
        ...
        CalcPseudoP_threaded();
    } else {
        double* values      = data1_vecs[0];      // :564
        double* local_moran = local_moran_vecs[0];
        ...
        bool flag = gpu_lisa(clPath.mb_str(), num_obs, permutations, last_seed_used,
                             values, local_moran, w, _sigLocal);
```

`gpu_lisa()` builds the permuted lag from `values`, i.e. from `data1`. For the **bivariate**
variant the CPU builds it from `data2` (`ComputeLarger()`, `LisaCoordinator.cpp:650-656`). The
**median** variant, **multiple time periods** and **undefined values** are likewise not
implemented by the kernel, yet all of them take this branch.

**Reachability.** GPU enabled, OpenCL device with `cl_khr_fp64`, kernel file found (see GPU-5).
This affects Windows and Intel-Mac users who turn the GPU on. Not applicable to libgeoda.

**Evidence.** `private/opencl_bivariate/` runs GeoDa's own `Algorithms/gpu_lisa.cpp` with the
**unmodified** `Algorithms/lisa_kernel.cl` (sha256
`e547bbe273e871c15a45a738747816ac3c3c3f96902bc33cba0dd45bb202ecc7`) under PoCL, with a single
substitution `CL_DEVICE_TYPE_GPU` → `CL_DEVICE_TYPE_ALL` so the fp64-capable CPU device can be
selected. The harness is validated by a scalar emulation of the kernel that reproduces its
output bit for bit (3085/3085 observations). Bivariate Local Moran on natregimes
(`hr90` against a rare 0/1 indicator, n = 3085, 999 permutations): **149 significant on the GPU
vs 1081 on the CPU, 32.7 % of observations flipping significance**, against a 1.3 % Monte Carlo
noise floor. Isolating the defect (same seed, same stream, only the lag source changed):
1081 vs 138. The size is data dependent — with `y` a reshuffle of `x` (identical marginal) only
1.65 % flip.

**Fixed on this branch**: `cfceee2b`. The GPU is used only when
`!isBivariate && !using_median && row_standardize && num_time_vals == 1 && !has_undefined[0]`;
`JCCoordinator::CalcPseudoP()` falls back to the CPU for periods with undefined values.

**Suggested upstream fix.** Either the same guard (small, safe) or kernels for the other
variants. The guard is the minimal correct change.

**Testing.** Run each variant with the GPU on and off; the guarded variants must produce
*identical* numbers, because both paths then execute the same CPU code.

---

## GPU-2 — the OpenCL Local Moran kernel does not implement the CPU algorithm

**Code.** `Algorithms/lisa_kernel.cl:136-210` on upstream `master`. Five differences from
`AbstractCoordinator::CalcPseudoP_range()` + `LisaCoordinator::ComputeLarger()`:

| # | kernel | CPU | effect |
|---|---|---|---|
| 1 | `newRandom = (int)rng_val;` (`:180`) | `(int)(rng_val<0?ceil(rng_val-0.5):floor(rng_val+0.5))` (`AbstractCoordinator.cpp:548`, the fix for issue #488) | different sample; observation `n−1` is essentially never drawn; infinite loop when an observation needs all others |
| 2 | `if (localMoranPermuted > local_moran[i])` (`:199`) | `>=` (`LisaCoordinator.cpp:676`) | ties undercount `countLarger`, which inflates significance |
| 3 | candidate accepted whenever `newRandom != i` (`:182`) | also requires `w[newRandom].Size() > 0` | isolates enter the null |
| 4 | `permutedLag /= numNeighbors` unconditionally (`:196`) | only `if (row_standardize)` | matches only because `row_standardize` is always true (§D) |
| 5 | `permutedLag` accumulated in draw order (`:190`) | `GeoDaSet::Pop()` returns the reverse order (`AbstractCoordinator.cpp:557`) | last bits differ, which decides exact ties |
| 6 | `num_nbrs[i] = w[i].Size()` (`gpu_lisa.cpp:66`) | self-link subtracted (`AbstractCoordinator.cpp:523-525`) | one extra drawn neighbour for kernel weights |

Plus `size_t rnd_numbers[123];` (`:172`) whose bound is patched at run time by
`boost::replace_all(src_code, "123", ...)` (`gpu_lisa.cpp:92`) — a *string* replacement that also
rewrites the `1234` in the comment on the same line.

**Measured effect** — the six code-level differences above were re-read from upstream `master`
for this document; the per-observation counts below are taken from the earlier harness in
`private/opencl_fix/` and were not re-run here (`./build.sh` regenerates them). Real OpenCL
execution on PoCL, per-observation
seeding on both sides, 999 permutations, observations whose p-value differs from the CPU):
Guerry 80/85; Guerry at 99,999 permutations 83/85; US Homicides 2834/3085; 50×50 lattice with
ties, isolates and a self-neighbour 2462/2500; dense weights (100 neighbours) — the original
kernel never terminates. Per-defect impact on US Homicides / the lattice: rounding 2836 / 2396,
`>` vs `>=` 8 / 2445, isolates 0 / 2333, self-neighbours 0 / 1, summation order 0 / 1152.

**Fixed on this branch**: `04529b11`; after the fix all of the above are 0/0.

---

## GPU-3 — `gpu_localjoincount()` never ran its kernel and returned p = 0 for every observation

**Code.** `Algorithms/localjc_kernel.cl:17` declares
`__kernel void localjc(const int n, const int permutations, const unsigned long last_seed,
const unsigned long num_vars, ...)`, and `Algorithms/gpu_lisa.cpp:440` passes

```cpp
    ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&num_vars);   // kernel wants 8 bytes
```

which is `CL_INVALID_ARG_SIZE` by the OpenCL specification. `ret` is only tested after the *last*
`clSetKernelArg()` (`:447`), so the error is discarded; argument 3 stays unset;
`clEnqueueNDRangeKernel()` fails with `CL_INVALID_KERNEL_ARGS` (return value unchecked at `:470`);
`clEnqueueReadBuffer()` (also unchecked) returns the zero-initialised scratch buffer, which is
copied over the caller's `p` for *every* row (`:477-479`); and the function returns `true`.
GeoDa then draws every location at p = 0, the most significant value there is.

**Verified** by an earlier agent on this branch by running the original host code and kernel
under PoCL (`private/opencl_fix/`, §1.10; re-read for this document, not re-run): PoCL returns −51 for that `clSetKernelArg`, and the measured
output is **p = 0 for all 85 Guerry observations** and for all 3085 natregimes observations.

**Reachability.** Local Join Count (univariate, bivariate, multivariate, co-location) and both
quantile LISA dialogs, with the GPU enabled and the kernel found. Not applicable to libgeoda.

**Fixed on this branch**: `04529b11` — `num_vars` is no longer a kernel argument (the kernel never
used it), the enqueue/read return values now decide the function's result, and the `p` buffer is
`CL_MEM_READ_WRITE` and pre-filled so that untouched entries survive.

---

## GPU-4 — the GPU branch ignores `reuse_last_seed == false`

**Code.** The fresh-seed step lives in the CPU function only,
`Explore/AbstractCoordinator.cpp:458`:

```cpp
    if (!reuse_last_seed) last_seed_used = time(0);
```

`LisaCoordinator::CalcPseudoP()` (`:556-601`) and `JCCoordinator::CalcPseudoP()`
(`MLJCCoordinator.cpp:454-505`) call `gpu_lisa()` / `gpu_localjoincount()` with `last_seed_used`
directly and never execute that line. `last_seed_used` is initialised to `123456789`
(`AbstractCoordinator.cpp:103`) or to `GdaConst::gda_user_seed`.

**Consequence.** With the seed checkbox *off* — the setting that is supposed to mean "use a new
random seed each run" — every GPU run returns exactly the same pseudo p-values. The user has no
indication of this. With the checkbox on, behaviour is correct.

**Reachability.** GPU enabled and the kernel found. CONFIRMED. Not applicable to libgeoda
(`LISA::CalcPseudoP()` has no GPU branch).

**Proposed fix.** `private/upstream_bugs/patches/bug-GPU6-reuse-last-seed.patch` adds the same
line at the top of the GPU branch (`<time.h>` is already included at
`Explore/LisaCoordinator.cpp:24`). The same one-liner is needed in `MLJCCoordinator.cpp`; it is
not in the patch because the file is already modified on this branch.

**Testing.** Run the same analysis twice with the seed checkbox off and assert that the p-value
vectors differ; with the checkbox on, assert they are identical.

---

## GPU-5 — the macOS bundle installs the kernels next to the binary, the code loads them from `Resources`

**Code.** `Explore/LisaCoordinator.cpp:570-575` and `Explore/MLJCCoordinator.cpp:488-493`:

```cpp
        wxString exePath = GenUtils::GetExeDir();
#ifdef __WXMAC__
        wxString clPath = exePath + "../Resources/lisa_kernel.cl";
```

`GenUtils::GetExeDir()` (`GenUtils.cpp:2336-2342`) returns the directory of the executable, i.e.
`GeoDa.app/Contents/MacOS/`, so the code looks in `GeoDa.app/Contents/Resources/`. The makefile
used by the release workflow puts it somewhere else —
`BuildTools/macosx/GNUmakefile:86`:

```make
	cp $(GeoDa_ROOT)/Algorithms/lisa_kernel.cl build/GeoDa.app/Contents/MacOS
```

and `localjc_kernel.cl` is **not copied at all**. `.github/workflows/osx_build.yml:137-138` runs
`make` and `make app`, i.e. this makefile, for both `macos-15` and `macos-15-intel`; the Xcode
projects put the files in a Resources phase and therefore agree with the code, but they are not
what the release is built with.

**Consequence.** `gpu_lisa()` opens the path with an unchecked `std::ifstream`
(`Algorithms/gpu_lisa.cpp:83`), gets an empty source, `clBuildProgram()` fails and the function
returns `false`, so GeoDa shows *"GeoDa can't configure GPU device. Default CPU solution will be
used instead."* and falls back to the CPU. Results stay correct; the GPU option simply cannot
work in a macOS release. (On Apple Silicon it could not work anyway: Apple's OpenCL has no
`cl_khr_fp64`.) A side effect worth stating explicitly: **C, GPU-1, GPU-2, GPU-3 and GPU-4 are
not reachable on a macOS release build** — they affect Windows and Intel-Mac OpenCL users.

**Proposed fix.** `private/upstream_bugs/patches/bug-GPU7-mac-kernel-path.patch` copies both
kernels to `Contents/Resources` and signs them there.

**Testing.** Build with `make app`, then
`ls GeoDa.app/Contents/Resources/{lisa,localjc}_kernel.cl`, and check that enabling the GPU no
longer raises the dialog on a machine with an fp64 OpenCL device.

---

## GPU-6 — dead `gpu_distmatrix()`: six arguments set on a four-argument kernel

**Code.** `Algorithms/distmat_kernel.cl:1` declares
`__kernel void euclidean_dist(const unsigned long rows, const unsigned long columns,
__global float *a, __global float *r)` — four parameters. `Algorithms/distmatrix.cpp:109-114`
sets six:

```cpp
    ret = clSetKernelArg(kernel, 0, sizeof(cl_ulong), &_rows);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &_columns);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_int),   &start);      // no such parameter
    ret = clSetKernelArg(kernel, 3, sizeof(cl_int),   &end);        // no such parameter
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem),   &a_mem_obj);  // kernel index 2
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem),   &r_mem_obj);  // kernel index 3
```

so the two buffers would go to non-existent indices and the kernel would never receive them.

**Reachability.** `gpu_distmatrix()` has no caller: the only occurrences in the tree are its
definition (`distmatrix.cpp:7,21`) and its declaration (`distmatrix.h:5`). `distmat_kernel.cl` is
listed in the Xcode projects and the Windows projects but is not copied by the macOS makefile.
**COSMETIC-OR-DEAD-CODE.**

**Proposed fix.** Delete `Algorithms/distmatrix.{cpp,h}` and `Algorithms/distmat_kernel.cl` and
their project references, or fix the indices if the function is meant to be revived.

---

## GPU-7 — the GPU seeding makes all observations share their Monte Carlo noise

**Code.** Upstream `Algorithms/lisa_kernel.cl` and `Algorithms/localjc_kernel.cl` start the random sequence of
observation `i` at

```c
    size_t seed_start = i + last_seed;
```

and then consume one key per draw (`ThomasWangHashDouble(seed_start++)`), rejected draws included. The CPU code does
something else: one counter per *thread*, started at `last_seed_used + a` for the thread's first observation `a` and
running on through all its observations (`Explore/AbstractCoordinator.cpp:470`, `:545`). The OpenCL fixes and the
Metal port on this branch kept the kernel's line, because their tests compare the kernels with the CPU *algorithm*
fed with the same keys.

**Consequence.** Observation `i` reads the keys `seed + i, seed + i + 1, ...`, about `P * k` of them; observation
`i + 1` reads the same keys shifted by one. Every observation within `P * k` index positions (4,000 for 999
permutations and 4 neighbours: usually the whole data set) is therefore compared with nearly the same random sample.
Each pseudo p-value is still valid on its own and nothing is biased, but the noise of different observations is no
longer independent and does not average out over the map.

**Evidence.** `dev-notes/repro/gpu7_shared_noise/` (tracked, standalone C++, 2 minutes): data without spatial
structure, fixed data, 200 seeds, 5 data sets per row, 999 permutations. Seed-to-seed standard deviation of the number
of observations with p <= 0.05, relative to independent noise:

| n | GPU seeding | desktop CPU (10 threads) | keys per (seed, observation, permutation) |
|--:|--:|--:|--:|
| 900 | 2.46 [2.18, 2.61] | 1.04 | 1.01 |
| 3,600 | 4.05 [3.94, 4.21] | 1.05 | 1.02 |
| 10,000 | 4.82 [4.54, 5.17] | 1.04 | 1.02 |
| 3,600, 3..8 neighbours | 3.63 [3.38, 3.80] | 1.07 | 0.98 |

In counts: at n = 3,600 about 357 observations are significant; from seed to seed that number moves by +-21 with the
GPU seeding and by +-5 with the CPU code. The excess grows with n. The GPU path is statistically worse than the CPU
path it replaces, although both run the same number of permutations.

**Reachability.** Every GPU run (OpenCL upstream once GPU-2/GPU-3 are fixed; Metal on this branch). CONFIRMED by
replay; not yet measured on the kernels themselves.

**Proposed fix.** Start every permutation `q` of observation `i` from its own key,
`hash(hash(seed + i) + q)`, and keep everything else (draw, rejection, comparison). Permutations become independent
of each other, which also allows splitting a long run into several GPU dispatches without changing the result. The
CPU code is not touched: its results stay as they are. GPU results for a given seed change; the GPU path is off by
default and the OpenCL path did not work before this branch, so nobody depends on those numbers.

**Testing.** The host reference in `Algorithms/test_metal_lisa.mm` replays the kernel's keys, so it changes with the
kernel and the GPU == reference tests stay exact. New: the result must not depend on how the permutations are split
into dispatches; the shared-noise measurement repeated on the real kernels, before and after.

---

## LG-1 — libgeoda counts a self-link in the permutation size but not in the observed statistic

**This is the entry that matters most for rgeoda/pygeoda: it is reachable today, with default
arguments, and it makes p-values anti-conservative.**

**What happens.** The observed local statistic excludes the location itself from its
neighbourhood; the permutation draws one neighbour *more* than that, because the neighbour count
it uses is the raw list length.

**Code.** `sa/LISA.cpp:577` (and the `"lookup"` twin at `:470`):

```cpp
        // get full neighbors even if has undefined value
        int numNeighbors = weights->GetNbrSize(cnt);
        if (numNeighbors == 0) { ... }
```

`GalWeight::GetNbrSize()` is `return gal[obs_idx].Size();` (`weights/GalWeight.cpp:317`);
`GwtWeight::GetNbrSize()` likewise (`weights/GwtWeight.cpp:118`). Neither subtracts a self-link.
The observed statistic does subtract it — `sa/UniLocalMoran.cpp:66-71`:

```cpp
                for (size_t j=0; j<nbrs.size(); ++j) {
                    if (nbrs[j] != i &&  !undefs[nbrs[j]]) {   // not including the value at the location
                        sp_lag += data[ nbrs[j] ]; nn += 1;
                    }
                }
                sp_lag = sp_lag / nn;
```

The same pattern is in `sa/UniG.cpp`, `sa/UniGstar.cpp`, `sa/UniGeary.cpp`, `sa/UniJoinCount.cpp`.
GeoDa desktop does it correctly: `Explore/AbstractCoordinator.cpp:521-526` subtracts the
self-link before drawing.

**Reachability — wider than expected.** In libgeoda, **every kernel weights builder pushes the
self entry unconditionally**: libgeoda's `SpatialIndAlgs.cpp:299` (also `:464`, `:765`). The R argument
`use_kernel_diagonals` does **not** control whether the self-link exists — it only decides
whether the self entry's *weight value* is the kernel value or is pinned to 1.0
(libgeoda's `SpatialIndAlgs.cpp:238`). Measured at R level: `kernel_knn_weights(..., use_kernel_diagonals =
FALSE)` and `... = TRUE` both give 7 neighbours including self for all 85 Guerry observations and
produce **bit-identical** `local_moran` output across 5 seeds — for `local_moran` the flag is a
complete no-op. So `kernel_weights()` / `kernel_knn_weights()` at their R defaults already hit
this. Contiguity (queen/rook) and plain knn/distance weights carry no self-link and are
unaffected. GeoDa desktop is not affected (it subtracts the self-link) — though see §F for the
desktop's own self-link problem in the observed Local Geary.

**Reproduction 1 — C++, against the real libgeoda functions**
(`private/upstream_bugs/libgeoda_repro/`, 12 libgeoda translation units compiled read-only with
`-I<libgeoda> -D__USE_PTHREAD__ -std=c++17`, Guerry queen weights ± a self-link on every
observation, 999 permutations, seed 123456789, `nCPUs = 1`, `permutation_method = "complete"`):

* the observed LISA value is identical for 85/85 observations (max difference 0.0e+00), so only
  the null moves;
* 80/85 p-values change; **70 get smaller**, 10 larger; mean change −0.01628 (min −0.087,
  max +0.027); **5 observations cross into p ≤ 0.05, none out**;
* instrumented copy: the observed statistic averages `k` values and the permutation averages
  `k+1`, for 85/85 observations;
* the null is **not shifted** (mean −0.013464 → −0.013376) but **narrowed**: standard-deviation
  ratio 0.8923 measured, against the theoretical `sqrt(k/(k+1)) = 0.9034`.

A narrower null with the same centre means the observed value falls further into the tail than it
should: the test **over-rejects**.

**Reproduction 2 — R, through the installed package**
(`private/upstream_bugs/r_repro/`, rgeoda 0.1.1 built into a private library, Guerry,
`Crm_prs`, 999 permutations). `knn_weights(6)` (6 links, no self) versus
`kernel_knn_weights(6)` (the same 6 real neighbours plus a self-link — verified elementwise);
`lisa_values()` are identical to the last bit, `lisa_num_nbrs()` reports 6 vs 7:

```
count p<=0.05 over 20 seeds
  knn(6)         : mean 35.75  sd 0.64
  kernel_knn(6)  : mean 39.40  sd 0.94     <- same observed statistic, +3.65 significant
  knn(7)         : mean 31.65  sd 0.67     <- a genuine 7th neighbour moves the OTHER way
difference kernel_knn(6) - knn(6) positive in 20/20 seeds, paired t p = 2.95e-14
signed dp: mean -0.010869, 81.3% negative   (seed noise: mean -0.000278, 47.3% negative)
null calibration over 20 spatial reshuffles, rejection rate at alpha = 0.05:
  knn(6) 8.76%   vs   kernel_knn(6) 12.35%   (paired p = 5.5e-07)
```

Two honest caveats. (i) The per-pair magnitude is only ~1.5× the seed-noise magnitude; the
evidence is the *direction*, which is consistent in 20/20 seeds. (ii) Both arms over-reject a
nominal 5 % by about 2×, which is a property of GeoDa's folded pseudo p-value, not of this bug;
the finding is the **41 % relative** inflation between two runs that share the same observed
statistic and the same real neighbour set.

**Proposed fix.** `private/upstream_bugs/patches/bug-P-libgeoda-self-neighbour.patch` subtracts a
self-link in both `LISA::CalcPseudoP_range()` and `LISA::PermCalcPseudoP_range()`, and in
`UniJoinCount::CalcPseudoP_range()`, using the existing `GeoDaWeight::CheckNeighbor()`. This
changes results for every rgeoda user who uses kernel weights, so it needs a release note.

**Testing a fix.** The C++ harness: after the fix, `p(queen)` and `p(queen + self-link on every
observation)` must agree to Monte Carlo noise (the real neighbour sets are identical), and the
instrumented null standard deviations must match. At R level, the 20-seed table above must lose
its systematic direction.

---

## LG-2 — `row_standardize` is dead in libgeoda too, and `UniG`/`UniGstar` would degenerate

**Code.** `row_standardize` is hard-coded `true` in both `LISA` constructors
(`sa/LISA.cpp:79`, `:107`) and in `BatchLISA` (`sa/BatchLISA.cpp:43`). The setter
`LISA::SetRowStandardize()` (`sa/LISA.cpp:711`) has no caller anywhere in `rgeoda/src` outside
the vendored geoda tree, and no `gda_*` API function in `gda_sa.h` takes such a parameter.
CONFIRMED-LATENT.

**What the dead branch would do.** Forcing `SetRowStandardize(false)` in a harness (construct,
record, set the flag, call the public `Run()` again with the same `last_seed_used`, so the draws
are bit-identical), on Guerry with queen weights, 999 permutations:

| class | behaviour with `row_standardize == false` |
|---|---|
| `UniLocalMoran` | 27 significant → **0**; min p 0.001 → 0.130 (the §D mean-vs-sum mismatch; conservative) |
| `UniG` | **all 85 p-values become exactly 0.001** |
| `UniGstar` | **all 85 p-values become exactly 0.001** |

The two G classes fail by different mechanisms. `sa/UniG.cpp:127-135` leaves
`permutedG = permutedLag`, i.e. a *raw sum of x* (order 1–10), while the observed
`lisa_vec[i] = (mean of neighbours)/(sum_x − x_i)` is of order 1/n; the count saturates at
`permutations` and the fold sends it to 0. `sa/UniGstar.cpp:127-131` is
`double permutedG = 0; if (validNeighbors > 0 && row_standardize) { ... }` — with the flag false
`permutedG` is never assigned at all, every permuted value is exactly 0, and `countLarger` is 0
directly.

**Proposed fix.** Either delete the flag and its setter, or implement binary weights in all five
`PermLocalSA` families consistently with the observed statistic. Deleting is the smaller change
and removes a trap. No patch provided: it is an API decision.

---

## LG-3 — NaN observed statistic, and NaN contagion into the cluster map

**What happens.** Several `ComputeLoalSA()` implementations divide by the count of usable
neighbours with only `GetNbrSize(i) == 0` as a guard, so an observation whose *usable* neighbour
count is zero divides `0.0 / 0u` and produces NaN.

**Code.** `sa/UniLocalMoran.cpp:59-71` (`nn` counts `nbrs[j] != i && !undefs[nbrs[j]]`, then
`sp_lag = sp_lag / nn;`), `sa/UniG.cpp:62-82`, `sa/UniGeary.cpp:59,73-74`. `UniGstar` is safe
because it adds the location itself before dividing (`sa/UniGstar.cpp:74-78`).

**Two ways in:** an observation whose only neighbour is itself (kernel weights, see LG-1), and an
observation all of whose neighbours are undefined (missing data).

**Consequence** (measured in `private/upstream_bugs/libgeoda_repro/`): `permutedSA[i] >= NaN` is
false for all 999 draws, so `countLarger = 0` and the observation is reported at
**p = 1/1000, the most significant value the library can produce**, with a cluster label derived
from NaN comparisons:

| class | lisa value | p | cluster |
|---|---|---|---|
| `UniLocalMoran` | NaN | 0.001 | High-High |
| `UniG` | NaN | 0.001 | Low-Low |
| `UniGeary` | NaN | 0.001 | Negative |

Worse, `sa/UniG.cpp:88-97` averages `lisa_vec` into `mean_g` with no NaN test, so **one**
degenerate observation makes `mean_g` NaN, `lisa_vec[i] >= mean_g` is then false everywhere, and
**every observation in the map is relabelled Low-Low**: cluster vector `[2 2 2 2 2 2]` against a
control's `[2 2 2 2 1 1]`. `sa/UniGstar.cpp:86-93` has the same shape.

**Proposed fix.** Guard each division (`if (nn == 0) { cluster = NEIGHBORLESS; continue; }`) and
skip non-finite values when averaging into `mean_g`. No patch provided: the right category for
such an observation (neighbourless? undefined?) is a maintainer decision, and it interacts with
LG-1's fix, which removes one of the two ways in.

**Testing a fix.** The harness constructs both degenerate cases explicitly; assert that no
`lisa_values()` entry is NaN and that a single degenerate row does not change any other
observation's cluster.

---

## LG-4 — `permutation_method = "lookup"` draws isolates, `"complete"` does not

**Code.** The two methods are selected at `sa/LISA.cpp:276` and `:300`. The `"complete"` draw
(`sa/LISA.cpp:626`, and `:597` in the `__JSGEODA__` variant) rejects a candidate with no
neighbours:

```cpp
                if (newRandom != cnt && !workPermutation.Belongs(newRandom)
                    && weights->GetNbrSize(newRandom)>0) {
```

The `"lookup"` table builder `LISA::PermCreateRange()` (`sa/LISA.cpp:380-405`, test at `:396`)
has no such test:

```cpp
    int max_rand = num_obs-2; // when one observation is always removed
    ...
            if (!workPermutation->Belongs(newRandom) && newRandom < num_obs ) {
```

so a neighbourless observation can enter the shared permutation table and contribute its value to
the null (it is not in `undefs` for `UniLocalMoran`/`UniGeary`/`UniG`, only `UniJoinCount` marks
isolates undefined, `sa/UniJoinCount.cpp:51`).

**Reachability.** `rgeoda::local_moran(..., permutation_method = "lookup")` and the same argument
on every other `local_*` function; the R documentation presents `"complete"` and `"lookup"` as
interchangeable options. Data sets with isolates are common (islands under contiguity weights).

**Status.** CONFIRMED as a code-level inconsistency between two options documented as
equivalent. **Not quantified here** — no reproduction was run for this one; the size of the
difference depends on how many isolates there are and how extreme their values are.

**Proposed fix.** Add the `GetNbrSize(newRandom) > 0` test to `PermCreateRange()`, or document
that the two methods are not equivalent. Note a second, deliberate difference: `"lookup"` shares
one permutation table across all observations, so the p-values of different observations are
computed from the same draws — that is the point of the option and is not a defect, but it is
worth stating in the documentation alongside any multiple-testing procedure
(`lisa_fdr()`, `lisa_bo()`).

---

## LG-5 — REFUTED: LOSH's pairing of drawn observations with positional weights is correct

**The claim.** `sa/UniLOSH.cpp` multiplies the residual of the `cp`-th *drawn* observation by
`nbr_w[cp]`, the weight of the `cp`-th *original* neighbour of `cnt`, which the survey called
"an arbitrary pairing".

**Why it is wrong.** That *is* the textbook weighted conditional permutation: the weights
`w_ij` are a property of the location pair structure and are held fixed while the *values* are
permuted over locations. Two checks:

1. With queen weights the question does not arise: `GalWeight::SetNeighbors()`
   (`weights/GalWeight.cpp:248-268`) only ever calls the two-argument `SetNbr`, which hard-sets
   weight 1.0, so `GetNeighborWeights()` returns 1.000 for all 420 Guerry queen links.
2. Where weights genuinely differ (`GwtWeight` from knn / distance / kernel builders), randomly
   reshuffling `nbr_w` among the drawn observations — destroying exactly the pairing the claim
   objects to — changes nothing beyond Monte Carlo noise: over five shuffle seeds, mean p
   0.2350–0.2361 against libgeoda's 0.2341, 14–15 significant against 15, one significance flip.

The "equal 1/k weights" alternative implied by the claim is the one that would be wrong: it would
compare a weighted observed statistic against an unweighted null. **Not worth filing.**

---

## X — Local Moran / Local Geary discard non-binary weight values in the observed statistic

Not a defect, but it belongs in the register because it surprises people and because it
interacts with §F and §LG-1.

`LisaCoordinator::Calc()` (`Explore/LisaCoordinator.cpp:521-528`) and
`LocalGearyCoordinator::CalcLocalGeary()` (`:721`) hard-code `bool is_binary = true;`, which makes
`GalElement::SpatialLag()` take the branch that ignores `nbrWeight` entirely and returns the plain
arithmetic mean of the neighbours. libgeoda does the same by construction: `UniLocalMoran`,
`UniGeary`, `UniG` and `UniGstar` never read `GetNeighborWeights()`. So a user who loads a kernel
or inverse-distance weights file and runs Local Moran gets an **unweighted** mean of the
neighbours, the weight values being used only to decide who is a neighbour. (`UniLOSH` is the one
exception in libgeoda: it does use the weight values.)

This is consistent between the two code bases and is probably intended — GeoDa documents LISA as
using row-standardized weights — but it is nowhere stated in the UI or in the rgeoda
documentation. INTENDED-OR-DISPUTABLE; a documentation item rather than a code change.

---

## LOSH-1 — NOT UPSTREAM: the observed LOSH lag includes the location itself, the null never does

**Scope note.** `sa/UniLOSH.{cpp,h}` is **not** upstream libgeoda code. It was added by commit
`8521405` ("Implement Local Spatial Heteroscedasticity (LOSH)") on the fork's `add-losh` branch
and is exposed through `rgeoda::local_losh()`. It is recorded here because the same self-link
question as LG-1 arises, with a different answer, and because it is easier to fix before the code
lands upstream than after.

**What happens.** Unlike `UniLocalMoran`, `UniGeary`, `UniG` and `UniGstar`, the two observed
passes in `UniLOSH::ComputeLoalSA()` have **no `nbrs[j] != i` test**
(`sa/UniLOSH.cpp:80-84` for the local mean, `:118-130` for the lagged residual `H_i`):

```cpp
        for (int j=0; j<num_nbrs; ++j) {
            if (nbrs[j] < num_obs && !undefs[nbrs[j]]) {
                lag_e  += local_residuals[nbrs[j]] * nbr_w[j];
                w_sum  += nbr_w[j];
            }
        }
```

so with a self-link the observed `H_i` contains the location's own residual `e_i`. The
permutation (`sa/UniLOSH.cpp:170-196`) draws `numNeighbors = GetNbrSize(cnt)` observations and
explicitly rejects `newRandom == cnt` (`:175`), so `e_cnt` is never in the null. The number of
terms matches (both `k+1`), but the observed sum contains a term the null structurally cannot.
The `"lookup"` twin (`:237-243`) does the same via the `nb >= cnt ? nb+1 : nb` shift.

Whether `x̄_i` should include `x_i` is a modelling choice in the LOSH literature; whether the
null should be able to reproduce the observed statistic is not.

**Reachability.** `rgeoda::local_losh()` with any kernel weights (self-link always present, see
LG-1). Contiguity and plain knn weights are unaffected.

**Evidence.** Guerry, queen versus queen + a self-link on every observation, 999 permutations:
**12 significance flips at p ≤ 0.05**. Caveat: adding a self-link also changes the observed
statistic here (unlike in LG-1, where it did not), so that number mixes the two effects; it is an
upper bound on the permutation-inconsistency alone, not a clean isolation. A clean isolation was
not run.

**Proposed fix.** Decide the intended definition, then make the two sides agree: either skip
`nbrs[j] == i` in both observed passes and subtract the self-link from `numNeighbors` (matching
`UniLocalMoran` and the LG-1 patch), or keep the self term in the observed statistic and add
`e_cnt` to every permuted statistic. The first is the one that matches the rest of libgeoda.

---

## Suggested grouping into upstream pull requests

The register mixes independent concerns. A plausible slicing, smallest and most defensible first:

**PR 1 — libgeoda: self-link in the conditional permutation (LG-1).**
`sa/LISA.cpp`, `sa/UniJoinCount.cpp`. One patch, a clear before/after with the R-level table, and
a release note because kernel-weight results change. This is the only entry in the register that
is both reachable with default arguments and produces systematically biased p-values, so it
should not be bundled with anything else. LG-3's kernel-weight entry point disappears as a side
effect, but LG-3 should still be its own PR.

**PR 2 — libgeoda: degenerate neighbourhoods (LG-3).** Guard the divisions and the `mean_g`
average. Needs a maintainer decision on categorisation, so keep it separate from PR 1.

**PR 3 — libgeoda: dead `row_standardize` (LG-2, and the libgeoda half of D).** Either remove the
flag or implement it. API-level, no result changes if removed.

**PR 4 — libgeoda: `"lookup"` vs `"complete"` (LG-4).** Small, self-contained, needs numbers
first.

**PR 5 — desktop: OpenCL correctness (GPU-1, GPU-2, GPU-3, C, GPU-4, GPU-5).** These are one
story — "the GPU path does not compute what the CPU path computes, and on macOS it cannot even
start" — and four of the six are already implemented on this branch
(`cfceee2b`, `04529b11`, `30faf4ff`). GPU-4 and GPU-5 are one-liners that belong with them. This
is the PR the Metal work already implies; GPU-5 in particular should be mentioned because it
explains why nobody would have noticed the rest on a Mac.
**GPU-7 changes THEIR design, not a bug in the usual sense, so it needs its own argument:** put it in the
OpenCL PR as a separate commit (or a separate PR right after it) whose description is the GPU-7 entry: the
two-line explanation of the shifted keys, the table, and `dev-notes/repro/gpu7_shared_noise/shared_noise.cpp`
attached so that a maintainer can rerun it in two minutes without GeoDa. The Metal PR must use the same keys as the
OpenCL kernels, so the order is: OpenCL fixes incl. GPU-7 first, or both in one PR.

**PR 6 — desktop: time-period handling in the shared draw (A, B).** Same function, same reviewer
context, both need the maintainer to confirm the intended semantics. Do not bundle with anything
else, because A contains a judgement call.

**PR 7 — desktop: `row_standardize` dead branches (D, E).** Two one-line changes plus, ideally,
removal of the dead parameter and of the commented-out checkbox. No reachable result changes,
which makes it an easy review.

**PR 8 — desktop: self-links in Local Geary (F), and `GalElement::Update()` (J).** Both are about
`GalElement`/`SpatialLag` bookkeeping. F changes reported values for kernel weights, so it needs
its own discussion; consider splitting.

**PR 9 — dead code (G, I, L, M, N, GPU-6).** One cleanup PR, no behaviour change, easy to accept
or reject wholesale. Keep G's one-line fix in it even though the caller is commented out.

**Not to be filed:** H, K, LG-5 (refuted), X (documentation).
**Fork-internal:** LOSH-1, to be fixed before the LOSH work is proposed upstream.

---

## Open questions

1. **A — which weights should the shared draw use?** One permutation serves all periods, so no
   period's weights are the "right" ones. The patch picks period 0 to keep the single-period case
   bit-identical; a maintainer may prefer the unmodified weights. Needs an upstream decision
   before a PR.
2. **D / E / LG-2 — is `row_standardize` meant to come back?** All three would be trivial to fix
   *and* trivial to delete. If binary weights are wanted, the observed statistics have to change
   too, which is a feature request, not a bug fix.
3. **F — fix the second lag, or strip self-links at load time?**
   `GalElement::RemoveSelfNeighbor()` exists and is called from nowhere. Calling it once when
   weights are loaded for a LISA-type analysis would fix F, the desktop half of the self-link
   family, and the `Size()==1`-self degenerate case in one place — but it would change results for
   every kernel-weights user and it would contradict the "handle the diagonal correctly" logic
   that `SpatialLag(..., self_id)` was written for.
4. **LG-4 — how big is the `"lookup"`/`"complete"` difference?** Not measured. Needs a data set
   with isolates and extreme values at those isolates.
5. **GPU-2 numbers not re-run.** The per-defect impact counts are inherited from
   `private/opencl_fix/`; the code-level differences were re-verified for this document, the
   numbers were not. `private/opencl_fix/build.sh` regenerates them under PoCL.
6. **Thread-count dependence (not filed as a bug).**
   `AbstractCoordinator::CalcPseudoP_threaded()` gives thread `i` covering `[a,b]` the seed
   `last_seed_used + a` and then lets the stream run through the whole chunk
   (`Explore/AbstractCoordinator.cpp:460-471`), so **GeoDa's pseudo p-values depend on the number
   of CPU cores** even with a fixed user seed. libgeoda has the same structure
   (`sa/LISA.cpp:494-556`, and `cpu_threads` is an rgeoda argument). This is a reproducibility
   wart rather than a wrong number — every core count gives a valid Monte Carlo sample — but it
   defeats "use a specified seed" as a reproducibility guarantee across machines. Worth raising
   upstream as a question before proposing a change.
7. **Isolate policy differs between statistics** (not filed): `LisaCoordinator` folds isolates
   into the undefined mask and removes their links; `JCCoordinator`, `GStatCoordinator` and
   `LocalGearyCoordinator` do not. The Join Count draw rejects *undefined* candidates while the
   LISA draw rejects *neighbourless* ones. All defensible in isolation, but the inconsistency is
   undocumented.

---

## Reproduction index

| folder (untracked, `private/upstream_bugs/`) | covers | how to run |
|:---|:---|:---|
| `multitime/` | A, B | `clang++ -O2 -std=c++17 -ffp-contract=off -o multitime multitime.cpp && ./multitime` |
| `bugC/` | C | `clang++ -O2 -std=c++17 -ffp-contract=off -o bugC bugC.cpp && ./bugC` |
| `desktop_repro/` | D, E, F, G, J | `bash build.sh` |
| `libgeoda_repro/` | LG-1, LG-2, LG-3, LG-5, LOSH-1 | `bash build.sh` (compiles libgeoda read-only) |
| `r_repro/` | LG-1 at R level, D reachability | `Rscript 01_checks.R` etc., private library in `../Rlib` |
| `patches/` | proposed fixes, **not applied** | see `patches/README.md` |
| **tracked:** `dev-notes/repro/gpu7_shared_noise/` | GPU-7 | `clang++ -O2 -std=c++14 shared_noise.cpp -o shared_noise && ./shared_noise` |

All patches are written against **upstream `master`** (`f7696a4b`) for the geoda files and against
the libgeoda submodule as checked out here. Verified to apply with `git apply --check` from the
geoda repo root, except `bug-D-…`, `bug-GPU6-…` and `bug-GPU7-…`, whose target files are already
modified on this development branch; those three were verified with `patch -p1 --dry-run` against
a clean `git show master:` extraction, and `bug-P-…` against the libgeoda working tree.
| `../opencl_bivariate/`, `../opencl_fix/` | GPU-1, GPU-2, GPU-3 (earlier work, re-read but not re-run) | `./build.sh` (needs PoCL) |


## Addenda after the review of the Metal variants (2026-09-18)

- **BUG B is not a defect at all**: the variants reviewer proved that computing `numNeighbors` as a true maximum
  over periods is equivalent to the upstream loop for every input (mutant M10 of `private/variants_review/`
  survives every test because it is an identity). Nothing to fix.
- **OpenCL / Metal Join Count and isolates**: `JCCoordinator` marks isolates undefined (MLJCCoordinator.cpp:258-263),
  its CPU draw rejects them (:649) and they get no p-value (:623). The first OpenCL fix on this branch (04529b11)
  missed that; fixed in c0d260ac (OpenCL) and 16693d07 (Metal). Evidence: chicago carjackings with a distance band
  (4 isolates): 124 of 5,272 all-samples checks failed for OpenCL before, 0 after.
- **GPU-4 fixed** on this branch in 16693d07 (both coordinators draw `time(0)` when `reuse_last_seed` is false,
  before the GPU call, per time period in `JCCoordinator`).
- The new significance-category loop in `LisaCoordinator::CalcPseudoP()` also changes the OpenCL result for
  self-only observations on Windows/Linux (second half of BUG C): intended.
- Off macOS the OpenCL Join Count still has no undefined-values mask: the coordinator keeps such periods on the CPU.
