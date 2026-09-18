# GPU-7: reproduction

`shared_noise.cpp` is standalone C++ (no GeoDa, no GPU, no libraries). It replays GeoDa's conditional permutation
draw with the three ways of choosing the random keys and measures how stable the result is from seed to seed.

```bash
clang++ -O2 -std=c++14 shared_noise.cpp -o shared_noise   # g++ -O2 -std=c++14 -pthread on Linux
./shared_noise                                            # P = 999, 200 seeds, 5 data sets; about 2 minutes on 10 cores
./shared_noise 199 20 1                                   # under a second, same picture
```

Output of the default run (Apple M-series, 2026-09-18): [`results_P999.md`](results_P999.md).

How to read it: data have no spatial structure, so every difference between seeds is Monte Carlo noise. `ratio` is
the seed-to-seed standard deviation of the number of observations with p <= 0.05, divided by the value that
independent noise gives. The upstream GPU seeding gives 2.5 (n = 900), 4.1 (n = 3,600) and 4.8 (n = 10,000); the
desktop CPU code 1.04 to 1.07; keys per (seed, observation, permutation) 1.0. Means are equal: there is no bias.

Why: with `seed_start = i + last_seed` observation `i` reads the keys `seed + i, seed + i + 1, ...`, about `P * k` of
them. Observation `i + 1` reads the same keys shifted by one, so permutation `q` of one observation draws almost the
same neighbours as a permutation of every other observation within `P * k` index positions: all of them are compared
with nearly the same random sample, and their noise does not average out over the map.

Limits: a replay, not the kernels themselves (the upstream OpenCL kernel cannot be run as is, see GPU-2); one
statistic (Local Moran, row-standardized), iid normal data, 4 or 3..8 neighbours.
