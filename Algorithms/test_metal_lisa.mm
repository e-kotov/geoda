// Standalone test of the Apple Metal permutation kernels against the CPU algorithm:
//
//   clang++ -std=c++14 -O2 -fobjc-arc Algorithms/test_metal_lisa.mm \
//       -framework Metal -framework Foundation -o test_metal_lisa
//   ./test_metal_lisa Algorithms/lisa_kernel.metal Algorithms/localjc_kernel.metal
//
// The reference below is the conditional permutation of
// AbstractCoordinator::CalcPseudoP_range() / JCCoordinator::CalcPseudoP_range().
// 1. With the random sequence of observation i starting at seed + i (what the GPU
//    does), GPU and CPU draw the same permutations, so pseudo p-values must be equal.
//    The only exception are permutations tied with the observed Local Moran: there
//    the CPU result depends on roundoff, while the GPU never counts a tie as larger.
// 2. With one sequential random sequence (what single threaded GeoDa does), pseudo
//    p-values must agree within Monte Carlo error.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// wx-free stand-in for ShapeOperations/GalWeight.h
class GalElement {
    std::vector<long> nbr;
public:
    void SetSizeNbrs(size_t sz) { nbr.resize(sz); }
    void SetNbr(size_t pos, long n) { nbr[pos] = n; }
    long Size() const { return (long)nbr.size(); }
    long operator[](size_t n) const { return nbr[n]; }
    bool Check(long n) const { return std::find(nbr.begin(), nbr.end(), n) != nbr.end(); }
};

#define STANDALONE_TEST
#include "metal_lisa.mm"
#include "test_data_guerry.h"
#include "test_data_natregimes.h"

// Gda::ThomasWangHashDouble()
static double ThomasWangHashDouble(uint64_t key)
{
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return 5.42101086242752217E-20 * key;
}

// CPU reference for both Local Moran (is_jc = false, row-standardized weights) and
// Local Join Count (is_jc = true)
static void cpu_reference(bool is_jc, bool per_obs_seed, int n, int permutations, uint64_t seed,
                          const std::vector<double>& x, const std::vector<double>& local_sa,
                          const std::vector<GalElement>& w, std::vector<double>& p,
                          std::vector<double>* p_no_ties = 0, std::vector<int>* num_ties = 0)
{
    int max_rand = n - 1;
    uint64_t seed_start = seed;
    double max_abs = 1;
    size_t max_nbrs = 0;
    for (int i = 0; i < n; i++) {
        max_abs = std::max(max_abs, fabs(x[i]));
        max_nbrs = std::max(max_nbrs, (size_t)w[i].Size());
    }
    const double tie_tol = (float)(1e-11 * max_abs * (max_nbrs + 1));
    std::vector<bool> drawn(n, false);
    std::vector<int> perm_nbrs;

    for (int i = 0; i < n; i++) {
        if (per_obs_seed) seed_start = seed + i;
        if (is_jc && local_sa[i] == 0) { p[i] = 0; continue; }
        int num_nbrs = (int)w[i].Size();
        if (w[i].Check(i)) num_nbrs -= 1;
        if (num_nbrs == 0) continue;

        int count_larger = 0, count_larger_no_ties = 0, ties = 0;
        for (int perm = 0; perm < permutations; perm++) {
            perm_nbrs.clear();
            while ((int)perm_nbrs.size() < num_nbrs) {
                double rng_val = ThomasWangHashDouble(seed_start++) * max_rand;
                int r = (int)(rng_val < 0.0 ? ceil(rng_val - 0.5) : floor(rng_val + 0.5));
                if (r != i && !drawn[r] && (is_jc || w[r].Size() > 0)) {
                    drawn[r] = true;
                    perm_nbrs.push_back(r);
                }
            }
            double lag = 0;
            for (int j = num_nbrs - 1; j >= 0; j--) { // GeoDaSet::Pop() order
                lag += x[perm_nbrs[j]];
                drawn[perm_nbrs[j]] = false;
            }
            if (is_jc) {
                if (lag >= local_sa[i]) count_larger++;
            } else {
                double permuted = lag / num_nbrs * x[i];
                if (permuted > local_sa[i]) count_larger++;
                // same test with ties decided exactly
                // (difference of the sums of neighbors is a roundoff error, see metal_lisa())
                bool tie = x[i] == 0 || fabs(lag - local_sa[i] * num_nbrs / x[i]) <= tie_tol;
                if (tie) ties++;
                else if (permuted > local_sa[i]) count_larger_no_ties++;
            }
        }
        if (p_no_ties) {
            if (permutations - count_larger_no_ties <= count_larger_no_ties) {
                count_larger_no_ties = permutations - count_larger_no_ties;
            }
            (*p_no_ties)[i] = (count_larger_no_ties + 1.0) / (permutations + 1);
            (*num_ties)[i] = ties;
        }
        if (is_jc ? (permutations - count_larger < count_larger) : (permutations - count_larger <= count_larger)) {
            count_larger = permutations - count_larger;
        }
        p[i] = (count_larger + 1.0) / (permutations + 1);
    }
}

struct TestCase {
    const char* name;
    std::vector<double> x;   // standardized variable
    std::vector<int> zz;     // binary variable
    std::vector<GalElement> w;
};

static void standardize(std::vector<double>& x)
{
    double mean = 0, ssd = 0;
    for (double v : x) mean += v;
    mean /= x.size();
    for (double v : x) ssd += (v - mean) * (v - mean);
    double sd = sqrt(ssd / (x.size() - 1));
    if (!(sd > 0)) { printf("[FAIL] constant or invalid test data\n"); exit(1); }
    for (double& v : x) v = (v - mean) / sd;
}

// GeoDa sample data, see make_test_data.R
static TestCase sample_data(const char* name, int n, const double* x, const int* nbr_offset, const int* nbrs)
{
    TestCase tc;
    tc.name = name;
    tc.x.assign(x, x + n);
    standardize(tc.x);
    tc.w.resize(n);
    for (int i = 0; i < n; i++) {
        int k = nbr_offset[i + 1] - nbr_offset[i];
        tc.w[i].SetSizeNbrs(k);
        for (int j = 0; j < k; j++) tc.w[i].SetNbr(j, nbrs[nbr_offset[i] + j]);
    }
    for (double v : tc.x) tc.zz.push_back(v > 0);
    return tc;
}

// rook lattice with tied integer values, isolates and a self-neighbor
static TestCase lattice(int side)
{
    TestCase tc;
    tc.name = "50x50 lattice, rook, isolates + self-neighbor";
    int n = side * side;
    tc.w.resize(n);
    for (int i = 0; i < n; i++) {
        tc.x.push_back((double)(ThomasWangHashDouble(i) < 0.3 ? 5 : (i / side + i % side) % 7));
        int r = i / side, c = i % side;
        std::vector<long> nb;
        if (r > 0) nb.push_back(i - side);
        if (r < side - 1) nb.push_back(i + side);
        if (c > 0) nb.push_back(i - 1);
        if (c < side - 1) nb.push_back(i + 1);
        if (i == 7) nb.push_back(i);              // self-neighbor
        if (i % 97 == 0) nb.clear();              // isolate (asymmetric weights are fine here)
        tc.w[i].SetSizeNbrs(nb.size());
        for (size_t j = 0; j < nb.size(); j++) tc.w[i].SetNbr(j, nb[j]);
    }
    for (double v : tc.x) tc.zz.push_back(v == 5);
    standardize(tc.x);
    return tc;
}

// more than 64 neighbors (kernel work array is resized), unstandardized values of large
// magnitude with exact zeros and ties, one observation linked to all others
static TestCase dense(int n, int k)
{
    TestCase tc;
    tc.name = "dense weights, 100 neighbors, unstandardized";
    tc.w.resize(n);
    for (int i = 0; i < n; i++) {
        double v = floor(ThomasWangHashDouble(i + 1000) * 20) - 5;
        tc.x.push_back(v * 1e6);
        tc.zz.push_back(v > 8);
        int ki = (i == 0) ? n - 1 : k;
        tc.w[i].SetSizeNbrs(ki);
        for (int j = 0; j < ki; j++) tc.w[i].SetNbr(j, (i + 1 + j) % n);
    }
    return tc;
}

static int failures = 0;

static void check(bool ok, const char* what)
{
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
}

static void run(TestCase& tc, const char* lisa_path, const char* jc_path, int permutations)
{
    const uint64_t seed = 123456789; // GeoDa's default seed
    const double untouched = -1;
    int n = (int)tc.x.size();
    printf("%s (n=%d, permutations=%d)\n", tc.name, n, permutations);

    for (int is_jc = 0; is_jc <= 1; is_jc++) {
        // observed statistics
        std::vector<double> x = tc.x, local_sa(n, 0);
        if (is_jc) x.assign(tc.zz.begin(), tc.zz.end());
        for (int i = 0; i < n; i++) {
            double lag = 0;
            int num_nbrs = 0;
            for (long j = 0; j < tc.w[i].Size(); j++) {
                if (tc.w[i][j] != i) { lag += x[tc.w[i][j]]; num_nbrs++; }
            }
            if (num_nbrs > 0) local_sa[i] = is_jc ? x[i] * lag : x[i] * lag / num_nbrs;
        }

        std::vector<double> p_gpu(n, untouched), p_cpu(n, untouched), p_seq(n, untouched), p_no_ties(n, untouched);
        std::vector<int> num_ties(n, 0);
        bool ok = is_jc ? metal_localjoincount(jc_path, n, permutations, seed, 1, tc.zz.data(), local_sa.data(), tc.w.data(), p_gpu.data())
                        : metal_lisa(lisa_path, n, permutations, seed, x.data(), local_sa.data(), tc.w.data(), p_gpu.data());
        check(ok, is_jc ? "metal_localjoincount() runs" : "metal_lisa() runs");
        if (!ok) continue;
        cpu_reference(is_jc, true, n, permutations, seed, x, local_sa, tc.w, p_cpu, &p_no_ties, &num_ties);
        cpu_reference(is_jc, false, n, permutations, seed, x, local_sa, tc.w, p_seq);

        int n_diff = 0, n_diff_no_ties = 0, n_tied = 0;
        double max_diff_seq = 0;
        for (int i = 0; i < n; i++) {
            if (num_ties[i] > 0) n_tied++;
            else if (p_gpu[i] != p_cpu[i]) n_diff++;
            if (!is_jc && p_gpu[i] != p_no_ties[i]) n_diff_no_ties++;
            max_diff_seq = std::max(max_diff_seq, fabs(p_gpu[i] - p_seq[i]));
        }
        printf("  %d observations with tied permutations; sequential seed: max p-value diff %g\n", n_tied, max_diff_seq);
        if (is_jc) {
            check(n_diff == 0 && n_tied == 0, "identical to CPU");
        } else {
            check(n_diff == 0, "identical to CPU where no permutation is tied with the observed value");
            check(n_diff_no_ties == 0, "identical to CPU with ties decided exactly");
        }
        // difference of two independent Monte Carlo estimates, 6 standard errors
        check(max_diff_seq <= 6 * sqrt(0.5 / permutations), "agrees with sequentially seeded CPU within Monte Carlo error");
    }
}

int main(int argc, char* argv[])
{
    const char* lisa_path = (argc > 1) ? argv[1] : "Algorithms/lisa_kernel.metal";
    const char* jc_path = (argc > 2) ? argv[2] : "Algorithms/localjc_kernel.metal";

    if (!is_metal_supported()) {
        printf("[SKIP] No Metal device available.\n");
        return 0;
    }
    TestCase g = sample_data("Guerry donatns, queen", guerry_n, guerry_x, guerry_nbr_offset, guerry_nbrs);
    TestCase u = sample_data("US Homicides hr90, queen", natregimes_n, natregimes_x, natregimes_nbr_offset, natregimes_nbrs);
    TestCase l = lattice(50);
    run(g, lisa_path, jc_path, 999);
    run(g, lisa_path, jc_path, 99999);
    run(u, lisa_path, jc_path, 999);
    run(l, lisa_path, jc_path, 999);

    TestCase d = dense(200, 100);
    run(d, lisa_path, jc_path, 999);

    // an isolate leaves too few observations to draw 3 neighbors from: the CPU code
    // would never finish, the GPU code must refuse
    std::vector<GalElement> w(4);
    w[0].SetSizeNbrs(3);
    for (int j = 0; j < 3; j++) w[0].SetNbr(j, j + 1);
    w[1].SetSizeNbrs(1); w[1].SetNbr(0, 0);
    w[2].SetSizeNbrs(1); w[2].SetNbr(0, 0);
    double x4[4] = {1, -1, 2, 0}, lm4[4] = {0.3, -1, 2, 0}, p4[4];
    printf("too few observations to permute\n");
    check(!metal_lisa(lisa_path, 4, 99, 1, x4, lm4, w.data(), p4), "metal_lisa() returns false");

    printf(failures ? "%d CHECK(S) FAILED\n" : "ALL METAL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
