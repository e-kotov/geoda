#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// the CPU arithmetic must be reproduced exactly: no fused multiply-add
#pragma OPENCL FP_CONTRACT OFF

// Gda::ThomasWangHashDouble()
double ThomasWangHashDouble(ulong key);
double ThomasWangHashDouble(ulong key)
{
    key = (~key) + (key << 21); // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8); // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4); // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return 5.42101086242752217E-20 * key;
}

// Conditional permutation of JCCoordinator::CalcPseudoP_range() (no undefined
// values), with the random sequence of observation i starting at last_seed + i.
// num_nbrs[i] is the number of neighbors to permute, self excluded; 0 for isolates, which
// JCCoordinator marks undefined: never drawn, no p-value; -1 if the only neighbor is itself.
// MAX_NBRS is defined by the host: at least the largest number of neighbors.
__kernel void localjc(const int n, const int permutations, const unsigned long last_seed, __global int *zz,  __global double *local_jc,  __global int *num_nbrs, __global double *p) {

    // Get the index of the current element
    int i = get_global_id(0);

    if (i >= n) {
        return;
    }
    int numNeighbors = num_nbrs[i];
    if (numNeighbors == 0) {
        // isolate (JCCoordinator marks it undefined): leave p[i] as it was passed in
        return;
    }
    if (local_jc[i] == 0) {
        p[i] = 0;
        return;
    }
    if (numNeighbors < 0) {
        // only neighbor is itself: no permutation
        return;
    }

    ulong seed_start = i + last_seed;
    int max_rand = n-1;

    int j, perm, rand, newRandom;
    bool is_valid;
    double rng_val;
    double permutedLag;
    int countLarger = 0;

    // observations drawn in the current permutation, in draw order
    int rnd_numbers[MAX_NBRS];

    for (perm=0; perm<permutations; perm++ ) {
        rand=0;
        while (rand < numNeighbors) {
            // computing 'perfect' permutation of given size
            rng_val = ThomasWangHashDouble(seed_start++) * max_rand;
            // round is needed to fix issue
            // https://github.com/GeoDaCenter/geoda/issues/488
            newRandom = (int)(rng_val<0.0?ceil(rng_val - 0.5):floor(rng_val + 0.5));

            if (newRandom != i && num_nbrs[newRandom] != 0) {
                is_valid = true;
                for (j=0; j<rand; j++) {
                    if (newRandom == rnd_numbers[j]) {
                        is_valid = false;
                        break;
                    }
                }
                if (is_valid) {
                    rnd_numbers[rand] = newRandom;
                    rand++;
                }
            }
        }

        permutedLag = 0;
        for (j=numNeighbors-1; j>=0; j--) { // GeoDaSet::Pop() order
            permutedLag += zz[rnd_numbers[j]];
        }
        // binary weights
        if (permutedLag >= local_jc[i]) {
            countLarger++;
        }
    }

    // pick the smallest
    if (permutations-countLarger < countLarger) {
        countLarger = permutations-countLarger;
    }

    p[i] = (countLarger + 1.0)/(permutations+1.0);
}
