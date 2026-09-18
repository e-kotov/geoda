P = 999 permutations, 200 seeds per data set, 5 data sets per row (mean [min, max] over data sets)

| n | neighbours | scheme | N = #{p<=0.05} | sd(N) over seeds | sd if independent | ratio | corr adjacent | corr n/2 apart |
|--:|:--|:--|--:|--:|--:|:--|--:|--:|
| 900 | 4 | GPU: key = seed + i (upstream OpenCL kernel) | 93.8 | 6.27 | 2.54 | 2.46 [2.18, 2.61] | +0.078 | +0.059 |
| 900 | 4 | CPU: one counter per thread (desktop, 10 threads) | 93.9 | 2.64 | 2.54 | 1.04 [0.98, 1.07] | +0.002 | +0.059 |
| 900 | 4 | KEYED: hash(hash(seed + i) + permutation) (proposed) | 93.9 | 2.60 | 2.56 | 1.01 [0.96, 1.06] | +0.001 | -0.001 |
| 3600 | 4 | GPU: key = seed + i (upstream OpenCL kernel) | 357.2 | 21.24 | 5.24 | 4.05 [3.94, 4.21] | +0.066 | +0.047 |
| 3600 | 4 | CPU: one counter per thread (desktop, 10 threads) | 357.5 | 5.56 | 5.27 | 1.05 [0.99, 1.11] | +0.000 | +0.042 |
| 3600 | 4 | KEYED: hash(hash(seed + i) + permutation) (proposed) | 357.9 | 5.37 | 5.29 | 1.02 [0.94, 1.15] | +0.000 | -0.000 |
| 10000 | 4 | GPU: key = seed + i (upstream OpenCL kernel) | 995.2 | 42.77 | 8.87 | 4.82 [4.54, 5.17] | +0.072 | +0.001 |
| 10000 | 4 | CPU: one counter per thread (desktop, 10 threads) | 994.7 | 9.19 | 8.85 | 1.04 [0.97, 1.11] | -0.001 | +0.000 |
| 10000 | 4 | KEYED: hash(hash(seed + i) + permutation) (proposed) | 995.1 | 9.01 | 8.86 | 1.02 [0.98, 1.05] | +0.001 | -0.000 |
| 3600 | 3..8 random | GPU: key = seed + i (upstream OpenCL kernel) | 356.3 | 18.62 | 5.12 | 3.63 [3.38, 3.80] | +0.051 | +0.037 |
| 3600 | 3..8 random | CPU: one counter per thread (desktop, 10 threads) | 357.0 | 5.55 | 5.17 | 1.07 [1.03, 1.14] | +0.000 | +0.007 |
| 3600 | 3..8 random | KEYED: hash(hash(seed + i) + permutation) (proposed) | 356.9 | 5.11 | 5.19 | 0.98 [0.94, 1.03] | -0.001 | -0.001 |
