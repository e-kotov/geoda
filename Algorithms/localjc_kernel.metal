#include <metal_stdlib>
using namespace metal;

inline float wang_rnd(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return (float)seed / 4294967295.0f;
}

kernel void localjc_metal(
    constant int &n                       [[buffer(0)]],
    constant int &permutations            [[buffer(1)]],
    constant ulong &last_seed             [[buffer(2)]],
    constant ulong &num_vars              [[buffer(3)]],
    device const ushort *zz               [[buffer(4)]],
    device const ushort *local_jc         [[buffer(5)]],
    device const ushort *num_nbrs         [[buffer(6)]],
    device const ushort *nbr_idx          [[buffer(7)]],
    device float *p                       [[buffer(8)]],
    uint i                                [[thread_position_in_grid]])
{
    if (i >= (uint)n) {
        return;
    }
    if (local_jc[i] == 0) {
        p[i] = 0.0f;
        return;
    }

    uint numNeighbors = (uint)num_nbrs[i];
    if (numNeighbors == 0) {
        p[i] = 0.0f;
        return;
    }

    uint seed_start = (uint)(i + last_seed);
    int rnd_numbers[123];
    int countLarger = 0;
    float max_rand = (float)(n - 1);

    for (int perm = 0; perm < permutations; perm++) {
        int rand = 0;
        int permutedLag = 0;

        while (rand < (int)numNeighbors) {
            float rng_val = wang_rnd(seed_start++) * max_rand;
            int newRandom = (int)rng_val;

            if (newRandom != (int)i) {
                bool is_valid = true;
                for (int j = 0; j < rand; j++) {
                    if (rnd_numbers[j] == newRandom) {
                        is_valid = false;
                        break;
                    }
                }
                if (is_valid) {
                    permutedLag += (int)zz[newRandom];
                    if (rand < 123) {
                        rnd_numbers[rand] = newRandom;
                    }
                    rand++;
                }
            }
        }

        if (permutedLag >= (int)local_jc[i]) {
            countLarger++;
        }
    }

    if (permutations - countLarger < countLarger) {
        countLarger = permutations - countLarger;
    }

    p[i] = (float)(countLarger + 1) / (float)(permutations + 1);
}
