#include <metal_stdlib>
using namespace metal;

// Same draw as the CPU code: round(Gda::ThomasWangHashDouble(key) * max_rand),
// i.e. round(hash * max_rand / 2^64), in integer arithmetic (no fp64 on Apple GPUs)
inline int ThomasWangHashIndex(ulong key, ulong max_rand)
{
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (int)(mulhi(key, max_rand) + ((key * max_rand) >> 63));
}

// There is no fp64 on Apple GPUs: the host passes each double as a 128 bit fixed point
// number (two ulong: low, high; two's complement), so sums of neighbors are exact.
// local_moran_permuted >= local_moran (as in LisaCoordinator::ComputeLarger) is tested
// as sum of permuted neighbors vs lag_sum = the observed sum of neighbors. The CPU's
// double arithmetic errs by up to 2^-53 of the partial sums for each of its k additions
// (plus a few operations for the observed value): differences up to
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

kernel void lisa_metal(
    constant int &n                       [[buffer(0)]],
    constant int &permutations            [[buffer(1)]],
    constant ulong &last_seed             [[buffer(2)]],
    device const int *num_nbrs            [[buffer(3)]],
    device const Fixed128 *values         [[buffer(4)]],
    device const Fixed128 *lag_sum        [[buffer(5)]],
    device const int *value_sign          [[buffer(6)]],
    device int *count_larger              [[buffer(7)]],
    uint i                                [[thread_position_in_grid]])
{
    if (i >= (uint)n) {
        return;
    }

    int numNeighbors = num_nbrs[i];
    if (numNeighbors <= 0) {
        count_larger[i] = -1; // no permutation test
        return;
    }

    ulong seed_start = i + last_seed;
    ulong max_rand = (ulong)(n - 1);
    int countLarger = 0;

    // Observations drawn in the current permutation. MAX_NBRS is a power of 2, at least
    // twice the largest number of neighbors. Duplicates are found by scanning rnd_numbers
    // if there are at most 64 neighbors, otherwise with an open addressing
    // hash table
#if MAX_NBRS > 128
    int drawn[MAX_NBRS];
    int used_slots[MAX_NBRS / 2];
    for (int j = 0; j < MAX_NBRS; j++) drawn[j] = -1;
#else
    int rnd_numbers[MAX_NBRS / 2];
#endif

    for (int perm = 0; perm < permutations; perm++) {
        int rand = 0;
        Fixed128 permutedLag = {0, 0}, sumAbs = {0, 0};

        while (rand < numNeighbors) {
            int newRandom = ThomasWangHashIndex(seed_start++, max_rand);

            if (newRandom != (int)i && num_nbrs[newRandom] != 0) {
#if MAX_NBRS > 128
                int slot = newRandom & (MAX_NBRS - 1);
                while (drawn[slot] != -1 && drawn[slot] != newRandom) {
                    slot = (slot + 1) & (MAX_NBRS - 1);
                }
                bool is_valid = drawn[slot] == -1;
#else
                bool is_valid = true;
                for (int j = 0; j < rand; j++) {
                    if (newRandom == rnd_numbers[j]) {
                        is_valid = false;
                        break;
                    }
                }
#endif
                if (is_valid) {
                    permutedLag = add128(permutedLag, values[newRandom]);
                    sumAbs = add128(sumAbs, abs128(values[newRandom]));
#if MAX_NBRS > 128
                    drawn[slot] = newRandom;
                    used_slots[rand] = slot;
#else
                    rnd_numbers[rand] = newRandom;
#endif
                    rand++;
                }
            }
        }
#if MAX_NBRS > 128
        for (int j = 0; j < numNeighbors; j++) drawn[used_slots[j]] = -1;
#endif

        Fixed128 diff = add128(permutedLag, neg128(lag_sum[i]));
        // tol = (k + 4) * 2^-52 * (sumAbs + |lag_sum|)
        Fixed128 u = add128(sumAbs, abs128(lag_sum[i]));
        ulong c = (ulong)(numNeighbors + 4);
        Fixed128 v = {(u.lo >> 52) | (u.hi << 12), u.hi >> 52};
        Fixed128 tol = {v.lo * c, v.hi * c + mulhi(v.lo, c)};
        // diff >= -tol, or diff <= tol for negative values[i]
        bool larger = (value_sign[i] > 0 && (long)add128(diff, tol).hi >= 0) ||
                      (value_sign[i] < 0 && (long)add128(neg128(diff), tol).hi >= 0) ||
                      value_sign[i] == 0;
        if (larger) {
            countLarger++;
        }
    }

    if (permutations - countLarger <= countLarger) {
        countLarger = permutations - countLarger;
    }

    // pseudo p-value is computed on the host in double precision
    count_larger[i] = countLarger;
}
