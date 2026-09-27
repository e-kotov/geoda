#include <metal_stdlib>
using namespace metal;

// The 64 bit integer hash of Gda::ThomasWangHashDouble(), before it is scaled to a double
// (a bijection on the keys)
inline ulong ThomasWangHash(ulong key)
{
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

// Rounds the unsigned integer hi * 2^64 + lo to 53 significant bits, ties to even, as
// IEEE-754 binary64 round to nearest does. Requires fewer than 117 significant bits, so
// that the bits it drops are all in lo.
inline void round_to_53_bits(thread ulong &hi, thread ulong &lo)
{
    uint bits = hi != 0 ? 128 - (uint)clz(hi) : 64 - (uint)clz(lo);
    uint drop = bits > 53 ? bits - 53 : 0;          // < 64
    ulong unit = 1UL << drop;                        // weight of the last kept bit
    ulong rest = lo & (unit - 1);                    // the dropped bits
    ulong tie = unit >> 1;                           // 0 if nothing is dropped
    ulong odd = (lo >> drop) & 1;
    // up if rest > tie, or rest == tie and the kept part is odd
    ulong up = (ulong)(drop != 0 && rest + odd > tie);
    ulong kept = lo - rest;
    lo = kept + (up << drop);
    hi += (ulong)(lo < kept);
}

// The CPU draw, bit for bit: with h = ThomasWangHash(key), m = max_rand (1 <= m < 2^31),
// Gda::ThomasWangHashDouble(key) * max_rand rounded as (int)floor(v + 0.5) is
//   floor(fl(fl(fl(h) * 2^-64 * m) + 0.5))
// where fl() rounds to binary64, ties to even: (a) h to double, (b) the product with m
// (the factor 2^-64 is exact), (c) the addition of 0.5. There is no fp64 on Apple GPUs:
// the values are integers in units of 2^-64 (below 2^96), so each fl() is exactly
// round_to_53_bits() and floor() is the upper 64 bits.
inline int ThomasWangHashIndex(ulong key, ulong max_rand)
{
    ulong lo = ThomasWangHash(key), hi = 0;
    round_to_53_bits(hi, lo);                        // (a) fl(h), at most 2^64
    hi = hi * max_rand + mulhi(lo, max_rand);
    lo = lo * max_rand;                              // exact fl(h) * m, below 2^95
    round_to_53_bits(hi, lo);                        // (b)
    lo += 1UL << 63;                                 // + 0.5, exact
    hi += (ulong)(lo < (1UL << 63));
    round_to_53_bits(hi, lo);                        // (c)
    return (int)hi;
}

// Keys (GPU-7 in dev-notes/UPSTREAM_BUGS.md): permutation q of observation i draws from
// its own sequence of keys, which starts at ThomasWangHash(ThomasWangHash(last_seed + i) + q),
// all in 64 bit unsigned arithmetic. The key depends on (last_seed, i, q) only, so the
// result does not depend on the order in which the permutations are computed, and
// observations do not share their Monte Carlo noise.
inline ulong permutation_key(ulong obs_key, int q)
{
    return ThomasWangHash(obs_key + (ulong)q);
}

// There is no fp64 on Apple GPUs: the host passes each double as a 128 bit fixed point
// number (two ulong: low, high; two's complement), so sums of neighbors are exact.
// local_moran_permuted >= local_moran (as in LisaCoordinator::ComputeLarger) is tested
// as sum of permuted neighbors vs lag_sum = the observed lag times the number of valid
// neighbors. The CPU's double arithmetic errs by up to 2^-53 of the partial sums for each
// of its k additions (plus a few operations for the observed value): differences up to
// (k + 4) 2^-52 (sum of absolute values) cannot be told from ties by the CPU and are ties here.
struct Fixed128 {
    ulong lo;
    ulong hi;
};

inline Fixed128 add128(Fixed128 a, Fixed128 b)
{
    Fixed128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1 : 0);
    return r;
}

inline Fixed128 neg128(Fixed128 a)
{
    Fixed128 r;
    r.lo = ~a.lo + 1;
    r.hi = ~a.hi + (r.lo == 0 ? 1 : 0);
    return r;
}

inline Fixed128 abs128(Fixed128 a)
{
    return ((long)a.hi < 0) ? neg128(a) : a;
}

// a * m, exact while |a| * m stays below 2^126 (the host keeps |a| below 2^110)
inline Fixed128 mul128(Fixed128 a, ulong m)
{
    bool negative = (long)a.hi < 0;
    Fixed128 v = negative ? neg128(a) : a;
    Fixed128 r = { v.lo * m, v.hi * m + mulhi(v.lo, m) };
    return negative ? neg128(r) : r;
}

// (k + 4) * 2^-52 * u for a non-negative u: what the CPU's double arithmetic can lose
inline Fixed128 tol128(Fixed128 u, int k)
{
    Fixed128 shifted = { (u.lo >> 52) | (u.hi << 12), u.hi >> 52 };
    return mul128(shifted, (ulong)(k + 4));
}

inline bool nonneg128(Fixed128 a)
{
    return (long)a.hi >= 0;
}

// -1, 0 or 1 for a - b
inline int cmp128(Fixed128 a, Fixed128 b)
{
    if (a.hi != b.hi) return ((long)a.hi < (long)b.hi) ? -1 : 1;
    if (a.lo != b.lo) return (a.lo < b.lo) ? -1 : 1;
    return 0;
}

// metal_lisa() compiles the kernels with MAX_NBRS (a power of two, at least twice the
// largest number of neighbors), N_PERIODS (time periods that share one draw) and HAS_UNDEF.

// Observations drawn in the current permutation. Duplicates are found by scanning the
// drawn observations if there are at most 64 neighbors, otherwise with an open addressing
// hash table; drawn_obs() reads the j-th drawn observation back in both cases.
#if MAX_NBRS > 128
#define TABLE_SIZE MAX_NBRS
inline int drawn_obs(thread const int *table, thread const int *slots, int j) { return table[slots[j]]; }
#else
#define TABLE_SIZE 1
inline int drawn_obs(thread const int *table, thread const int *slots, int j) { return slots[j]; }
#endif

// The lag of one time period: the sum of the drawn values and of their absolute values
// (for the tie tolerance), and how many of the drawn observations are defined
struct Lag {
    Fixed128 sum;
    Fixed128 sumAbs;
    int valid;
};

inline void lag_add(thread Lag &lag, Fixed128 v)
{
    lag.sum = add128(lag.sum, v);
    lag.sumAbs = add128(lag.sumAbs, abs128(v));
    lag.valid++;
}

// Draws numNeighbors distinct observations other than self, as
// AbstractCoordinator::CalcPseudoP_range(): observations the CPU rejects (no neighbors in
// the last time period) are skipped. clear_permutation() prepares the next draw.
// FUSE sums the values as they are drawn, which is all a single time period without
// undefined values needs.
template <bool FUSE>
inline void draw_permutation(thread ulong &seed, int self, int numNeighbors, ulong max_rand,
                             device const uchar *draw_ok, thread int *table, thread int *slots,
                             device const Fixed128 *values, thread Lag &lag)
{
    int rand = 0;
    while (rand < numNeighbors) {
        int newRandom = ThomasWangHashIndex(seed++, max_rand);

        if (newRandom != self && draw_ok[newRandom] != 0) {
#if MAX_NBRS > 128
            int slot = newRandom & (MAX_NBRS - 1);
            while (table[slot] != -1 && table[slot] != newRandom) {
                slot = (slot + 1) & (MAX_NBRS - 1);
            }
            if (table[slot] == -1) {
                table[slot] = newRandom;
                slots[rand++] = slot;
                if (FUSE) lag_add(lag, values[newRandom]);
            }
#else
            bool is_valid = true;
            for (int j = 0; j < rand; j++) {
                if (newRandom == slots[j]) {
                    is_valid = false;
                    break;
                }
            }
            if (is_valid) {
                slots[rand++] = newRandom;
                if (FUSE) lag_add(lag, values[newRandom]);
            }
#endif
        }
    }
}

inline void clear_permutation(int numNeighbors, thread int *table, thread int *slots)
{
#if MAX_NBRS > 128
    for (int j = 0; j < numNeighbors; j++) table[slots[j]] = -1;
#endif
}

// Local Moran of every variant except the median: univariate (values are data1),
// bivariate (data2), undefined neighbors (left out of the lag, which changes the number
// of valid neighbors per permutation) and several time periods sharing one draw.
kernel void lisa_metal(
    constant int &n                       [[buffer(0)]],
    constant int &permutations            [[buffer(1)]],
    constant ulong &last_seed             [[buffer(2)]],
    device const int *num_nbrs            [[buffer(3)]],
    device const uchar *draw_ok           [[buffer(4)]],
    device const uchar *undef             [[buffer(5)]],
    device const uchar *count_empty       [[buffer(6)]],
    device const int *value_sign          [[buffer(7)]],
    device const Fixed128 *values         [[buffer(8)]],
    device const Fixed128 *lag            [[buffer(9)]],
    device int *count_larger              [[buffer(10)]],
    uint i                                [[thread_position_in_grid]])
{
    if (i >= (uint)n) {
        return;
    }

    int numNeighbors = num_nbrs[i];
    if (numNeighbors <= 0) {
        for (int t = 0; t < N_PERIODS; t++) count_larger[t * n + i] = -1; // no permutation test
        return;
    }

    ulong obs_key = ThomasWangHash((ulong)i + last_seed);
    ulong max_rand = (ulong)(n - 1);
    int countLarger[N_PERIODS];
    for (int t = 0; t < N_PERIODS; t++) countLarger[t] = 0;

    int table[TABLE_SIZE];
    int slots[MAX_NBRS / 2];
#if MAX_NBRS > 128
    for (int j = 0; j < TABLE_SIZE; j++) table[j] = -1;
#endif

    // one time period without undefined values sums every drawn value: the draw does it
#if N_PERIODS == 1 && !HAS_UNDEF
    const bool fuse = true;
#else
    const bool fuse = false;
#endif

#if !HAS_UNDEF
    // every permutation has the same number of valid neighbors, so the sum the observed
    // lag stands for is the same too
    Fixed128 observed[N_PERIODS];
    for (int t = 0; t < N_PERIODS; t++) observed[t] = mul128(lag[t * n + i], (ulong)numNeighbors);
#endif

    for (int perm = 0; perm < permutations; perm++) {
        Lag drawn_lag = { {0, 0}, {0, 0}, 0 };
        ulong seed = permutation_key(obs_key, perm);
        draw_permutation<fuse>(seed, (int)i, numNeighbors, max_rand, draw_ok,
                               table, slots, values, drawn_lag);

        // every time period reuses the same draw
        for (int t = 0; t < N_PERIODS; t++) {
            int base = t * n;
            Lag lag_t = { {0, 0}, {0, 0}, 0 };
            if (fuse) {
                lag_t = drawn_lag;
            } else {
                for (int j = 0; j < numNeighbors; j++) {
                    int nb = drawn_obs(table, slots, j);
#if HAS_UNDEF
                    if (undef[base + nb] != 0) {
                        continue; // undefined neighbors are left out of the lag
                    }
#endif
                    lag_add(lag_t, values[base + nb]);
                }
            }
#if HAS_UNDEF
            if (lag_t.valid == 0) {
                // the CPU does not divide and compares 0 with the observed Local Moran
                countLarger[t] += count_empty[base + i];
                continue;
            }
#endif
            // the observed lag is a mean: the sum it is compared with has as many terms
            const int validNeighbors = lag_t.valid;
#if HAS_UNDEF
            Fixed128 lag_sum = mul128(lag[base + i], (ulong)validNeighbors);
#else
            Fixed128 lag_sum = observed[t];
#endif
            Fixed128 diff = add128(lag_t.sum, neg128(lag_sum));
            Fixed128 tol = tol128(add128(lag_t.sumAbs, abs128(lag_sum)), validNeighbors);
            // diff >= -tol, or diff <= tol for negative values[i]
            int sign = value_sign[base + i];
            if ((sign > 0 && nonneg128(add128(diff, tol))) ||
                (sign < 0 && nonneg128(add128(neg128(diff), tol))) || sign == 0) {
                countLarger[t]++;
            }
        }
        clear_permutation(numNeighbors, table, slots);
    }

    for (int t = 0; t < N_PERIODS; t++) {
        int c = countLarger[t];
        if (permutations - c <= c) {
            c = permutations - c;
        }
        // pseudo p-value is computed on the host in double precision
        count_larger[t * n + i] = c;
    }
}

// The two values GenUtils::Median() averages, i.e. the drawn values of rank (k-1)/2 and
// k/2 (the same one when k is odd).  Quickselect with a Hoare partition, iterative
// because a GPU thread has no stack to recurse on, over the array of indices: O(k)
// comparisons on average instead of the O(k^2) of a counting selection, which dominated
// the median kernel at the neighbor counts distance-band weights produce.  Equal values
// make the partition walk inward from both ends, so tied data stays balanced.
// idx[] is scratch and comes back reordered.
inline void median2_of(thread int *idx, int k, device const Fixed128 *values,
                       thread Fixed128 &v_lo, thread Fixed128 &v_hi)
{
    const int hi_rank = k / 2;
    int lo = 0, hi = k - 1;
    while (lo < hi) {
        Fixed128 pivot = values[idx[(lo + hi) / 2]];
        int a = lo, b = hi;
        while (a <= b) {
            while (cmp128(values[idx[a]], pivot) < 0) a++;
            while (cmp128(values[idx[b]], pivot) > 0) b--;
            if (a <= b) {
                int t = idx[a]; idx[a] = idx[b]; idx[b] = t;
                a++; b--;
            }
        }
        if (hi_rank <= b) hi = b;
        else if (hi_rank >= a) lo = a;
        else break;                      // b < hi_rank < a: that element is in place
    }
    v_hi = values[idx[hi_rank]];
    v_lo = v_hi;
    if ((k & 1) == 0) {
        // even count: the other middle value is the largest of the lower half
        v_lo = values[idx[0]];
        for (int j = 1; j < hi_rank; j++)
            if (cmp128(values[idx[j]], v_lo) > 0) v_lo = values[idx[j]];
    }
}

// Median Local Moran: the permuted statistic is the median of the drawn values instead of
// their mean, so it is the sum of the one or two middle values (kept doubled, which makes
// the mean of two exact). lag[] holds the same sum for the observed neighbors, and only
// the CPU's own rounding of the two averages and products has to be tolerated.
kernel void lisa_median_metal(
    constant int &n                       [[buffer(0)]],
    constant int &permutations            [[buffer(1)]],
    constant ulong &last_seed             [[buffer(2)]],
    device const int *num_nbrs            [[buffer(3)]],
    device const uchar *draw_ok           [[buffer(4)]],
    device const uchar *undef             [[buffer(5)]],
    device const uchar *count_empty       [[buffer(6)]],
    device const int *value_sign          [[buffer(7)]],
    device const Fixed128 *values         [[buffer(8)]],
    device const Fixed128 *lag            [[buffer(9)]],
    device int *count_larger              [[buffer(10)]],
    uint i                                [[thread_position_in_grid]])
{
    if (i >= (uint)n) {
        return;
    }

    int numNeighbors = num_nbrs[i];
    if (numNeighbors <= 0) {
        for (int t = 0; t < N_PERIODS; t++) count_larger[t * n + i] = -1; // no permutation test
        return;
    }

    ulong obs_key = ThomasWangHash((ulong)i + last_seed);
    ulong max_rand = (ulong)(n - 1);
    int countLarger[N_PERIODS];
    for (int t = 0; t < N_PERIODS; t++) countLarger[t] = 0;

    int table[TABLE_SIZE];
    int slots[MAX_NBRS / 2];
    int defined[MAX_NBRS / 2];
#if MAX_NBRS > 128
    for (int j = 0; j < TABLE_SIZE; j++) table[j] = -1;
#endif

    for (int perm = 0; perm < permutations; perm++) {
        Lag unused = { {0, 0}, {0, 0}, 0 };
        ulong seed = permutation_key(obs_key, perm);
        draw_permutation<false>(seed, (int)i, numNeighbors, max_rand, draw_ok,
                                table, slots, values, unused);

        for (int t = 0; t < N_PERIODS; t++) {
            int base = t * n;
            int k = 0;
            for (int j = 0; j < numNeighbors; j++) {
                int nb = drawn_obs(table, slots, j);
#if HAS_UNDEF
                if (undef[base + nb] != 0) {
                    continue; // GenUtils::Median() of the defined neighbors only
                }
#endif
                defined[k++] = base + nb;
            }
#if HAS_UNDEF
            if (k == 0) {
                // GenUtils::Median() of nothing is 0, which the CPU compares with the
                // observed Local Moran
                countLarger[t] += count_empty[base + i];
                continue;
            }
#endif
            Fixed128 v1 = {0, 0}, v2 = {0, 0};
            median2_of(defined, k, values, v1, v2);
            Fixed128 median2 = add128(v1, v2);                    // median * 2
            Fixed128 diff = add128(median2, neg128(lag[base + i]));
            Fixed128 sumAbs = add128(abs128(v1), abs128(v2));
            Fixed128 tol = tol128(add128(sumAbs, abs128(lag[base + i])), 2);
            int sign = value_sign[base + i];
            if ((sign > 0 && nonneg128(add128(diff, tol))) ||
                (sign < 0 && nonneg128(add128(neg128(diff), tol))) || sign == 0) {
                countLarger[t]++;
            }
        }
        clear_permutation(numNeighbors, table, slots);
    }

    for (int t = 0; t < N_PERIODS; t++) {
        int c = countLarger[t];
        if (permutations - c <= c) {
            c = permutations - c;
        }
        count_larger[t * n + i] = c;
    }
}
