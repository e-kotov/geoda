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

kernel void localjc_metal(
    constant int &n                       [[buffer(0)]],
    constant int &permutations            [[buffer(1)]],
    constant ulong &last_seed             [[buffer(2)]],
    device const int *num_nbrs            [[buffer(3)]],
    device const int *zz                  [[buffer(4)]],
    device const int *local_jc            [[buffer(5)]],
    device int *count_larger              [[buffer(6)]],
    uint i                                [[thread_position_in_grid]])
{
    if (i >= (uint)n) {
        return;
    }

    int numNeighbors = num_nbrs[i];
    if (local_jc[i] == 0 || numNeighbors == 0) {
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
        int permutedLag = 0;

        while (rand < numNeighbors) {
            int newRandom = ThomasWangHashIndex(seed_start++, max_rand);

            if (newRandom != (int)i) {
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
                    permutedLag += zz[newRandom];
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

        if (permutedLag >= local_jc[i]) {
            countLarger++;
        }
    }

    if (permutations - countLarger < countLarger) {
        countLarger = permutations - countLarger;
    }

    // pseudo p-value is computed on the host in double precision
    count_larger[i] = countLarger;
}
