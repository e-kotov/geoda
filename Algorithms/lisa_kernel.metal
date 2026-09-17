#include <metal_stdlib>
using namespace metal;

inline float ThomasWangHashFloat(ulong key)
{
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return (float)(key & 0xFFFFFFFF) * 2.3283064365386963e-10f;
}

kernel void lisa_metal(
    constant int &n                       [[buffer(0)]],
    constant int &permutations            [[buffer(1)]],
    constant ulong &last_seed             [[buffer(2)]],
    device const float *values            [[buffer(3)]],
    device const float *local_moran       [[buffer(4)]],
    device const int *num_nbrs            [[buffer(5)]],
    device const int *nbr_idx             [[buffer(6)]],
    device float *p                       [[buffer(7)]],
    uint i                                [[thread_position_in_grid]])
{
    if (i >= (uint)n) {
        return;
    }

    int numNeighbors = num_nbrs[i];
    if (numNeighbors == 0) {
        p[i] = 1.0f;
        return;
    }

    ulong seed_start = i + last_seed;
    float max_rand = (float)(n - 1);
    int countLarger = 0;

    int rnd_numbers[123];

    for (int perm = 0; perm < permutations; perm++) {
        int rand = 0;
        float permutedLag = 0.0f;

        while (rand < numNeighbors) {
            float rng_val = ThomasWangHashFloat(seed_start++) * max_rand;
            int newRandom = (int)rng_val;

            if (newRandom != (int)i) {
                bool is_valid = true;
                for (int j = 0; j < rand; j++) {
                    if (newRandom == rnd_numbers[j]) {
                        is_valid = false;
                        break;
                    }
                }
                if (is_valid) {
                    permutedLag += values[newRandom];
                    if (rand < 123) {
                        rnd_numbers[rand] = newRandom;
                    }
                    rand++;
                }
            }
        }

        permutedLag /= (float)numNeighbors;
        float localMoranPermuted = permutedLag * values[i];
        if (localMoranPermuted > local_moran[i]) {
            countLarger++;
        }
    }

    if (permutations - countLarger <= countLarger) {
        countLarger = permutations - countLarger;
    }

    p[i] = (float)(countLarger + 1) / (float)(permutations + 1);
}
