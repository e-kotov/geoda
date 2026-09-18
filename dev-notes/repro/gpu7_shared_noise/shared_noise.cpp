// GPU-7: the GPU seeding of the conditional permutation test makes all observations share their Monte Carlo noise.
//
// Standalone, no dependencies:  clang++ -O2 -std=c++14 shared_noise.cpp -o shared_noise && ./shared_noise
// (g++ works too; add -pthread on Linux). Arguments (optional): P  seeds  datasets   (default 999 200 5)
//
// What is replayed, line by line from GeoDa:
//  * the draw: rng_val = ThomasWangHashDouble(key++) * (n-1); newRandom = round(rng_val); reject self and duplicates
//    (Explore/AbstractCoordinator.cpp:545-560, Algorithms/lisa_kernel.cl: same loop)
//  * the pseudo p-value: (min(countLarger, P - countLarger) + 1) / (P + 1)
//  * three ways to choose the key a draw starts from:
//    GPU    Algorithms/lisa_kernel.cl (upstream master):  size_t seed_start = i + last_seed;   one counter per
//           observation i, so observation i+1 reads the keys of observation i shifted by one
//    CPU    Explore/AbstractCoordinator.cpp:470:  uint64_t seed_start = last_seed_used + a;   one counter per THREAD,
//           running on through all observations a..b of the thread
//    KEYED  proposed: every permutation q of observation i starts from hash(hash(seed + i) + q)
//
// Experiment: iid N(0,1) data (NO spatial structure; the observed lag is the mean of k random other values), data
// fixed, the test repeated with many seeds. With healthy random numbers the noise of different observations is
// independent, and the seed-to-seed variance of N = #{p <= 0.05} is the sum of the per-observation variances.
// ratio = sd(N over seeds) / that independent-noise sd. 1 is ideal; > 1 means the map is less stable between seeds
// than the number of permutations promises. corr = correlation over seeds of the p-value noise of two observations.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

static inline uint64_t tw(uint64_t key) // Gda::ThomasWangHashDouble() before its scaling to [0, 1)
{
    key = (~key) + (key << 21); key ^= key >> 24; key = (key + (key << 3)) + (key << 8); key ^= key >> 14;
    key = (key + (key << 2)) + (key << 4); key ^= key >> 28; key += key << 31;
    return key;
}
static inline int draw(uint64_t key, int n) { return (int)std::floor(5.42101086242752217E-20 * tw(key) * (n - 1) + 0.5); }

enum Scheme { GPU = 0, CPU = 1, KEYED = 2 };
static const char* NAMES[] = {"GPU: key = seed + i (upstream OpenCL kernel)", "CPU: one counter per thread (desktop, 10 threads)",
                              "KEYED: hash(hash(seed + i) + permutation) (proposed)"};

struct Data { int n; std::vector<double> z, observed; std::vector<int> k; };

static Data make_data(int n, bool vary_k, uint64_t data_seed)
{
    Data d; d.n = n; d.z.resize(n); d.observed.resize(n); d.k.resize(n);
    std::mt19937_64 g(data_seed); std::normal_distribution<double> N01;
    for (int i = 0; i < n; i++) d.z[i] = N01(g);
    for (int i = 0; i < n; i++) {
        d.k[i] = vary_k ? 3 + (int)(g() % 6) : 4;
        std::vector<int> nb; double s = 0;
        while ((int)nb.size() < d.k[i]) {
            int j = (int)(g() % n);
            if (j != i && std::find(nb.begin(), nb.end(), j) == nb.end()) { nb.push_back(j); s += d.z[j]; }
        }
        d.observed[i] = s / d.k[i];
    }
    return d;
}

static void run(const Data& d, Scheme scheme, int P, uint64_t seed, std::vector<double>& p)
{
    const int n = d.n, T = 10;
    uint64_t running = 0;
    int nb[8];
    for (int i = 0; i < n; i++) {
        if (scheme == CPU && i % (n / T) == 0 && i / (n / T) < T) running = seed + i; // first observation of a thread
        uint64_t key = scheme == CPU ? running : seed + i;
        int larger = 0;
        for (int q = 0; q < P; q++) {
            if (scheme == KEYED) key = tw(tw(seed + i) + (uint64_t)q);
            int got = 0; double s = 0;
            while (got < d.k[i]) {
                int j = draw(key++, n);
                bool ok = j != i;
                for (int t = 0; t < got && ok; t++) ok = nb[t] != j;
                if (ok) { nb[got++] = j; s += d.z[j]; }
            }
            if (d.z[i] * s / d.k[i] >= d.z[i] * d.observed[i]) larger++;
        }
        if (scheme == CPU) running = key;
        p[i] = (std::min(larger, P - larger) + 1.0) / (P + 1.0);
    }
}

struct Result { double ratio, sdN, sdInd, meanN, corr_adj, corr_far; };

static Result measure(const Data& d, Scheme scheme, int P, int M)
{
    const int n = d.n;
    std::vector<std::vector<double> > p(M, std::vector<double>(n));
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < hw; w++) pool.emplace_back([&, w]() {
        for (int m = (int)w; m < M; m += (int)hw) run(d, scheme, P, 123456789ULL + 1000003ULL * m, p[m]);
    });
    for (auto& t : pool) t.join();

    std::vector<double> mean(n, 0), f(n, 0), N(M, 0);
    for (int m = 0; m < M; m++) for (int i = 0; i < n; i++) {
        mean[i] += p[m][i] / M;
        if (p[m][i] <= 0.05) { N[m] += 1; f[i] += 1.0 / M; }
    }
    auto corr = [&](int lag) {
        double xy = 0, xx = 0, yy = 0;
        for (int m = 0; m < M; m++) for (int i = 0; i + lag < n; i++) {
            double a = p[m][i] - mean[i], b = p[m][i + lag] - mean[i + lag];
            xy += a * b; xx += a * a; yy += b * b;
        }
        return xy / std::sqrt(xx * yy);
    };
    Result r; r.meanN = 0; for (double v : N) r.meanN += v / M;
    double var = 0; for (double v : N) var += (v - r.meanN) * (v - r.meanN) / (M - 1);
    double ind = 0; for (int i = 0; i < n; i++) ind += f[i] * (1 - f[i]);
    r.sdN = std::sqrt(var); r.sdInd = std::sqrt(ind); r.ratio = r.sdN / r.sdInd;
    r.corr_adj = corr(1); r.corr_far = corr(n / 2);
    return r;
}

int main(int argc, char** argv)
{
    const int P = argc > 1 ? atoi(argv[1]) : 999, M = argc > 2 ? atoi(argv[2]) : 200, D = argc > 3 ? atoi(argv[3]) : 5;
    printf("P = %d permutations, %d seeds per data set, %d data sets per row (mean [min, max] over data sets)\n\n", P, M, D);
    printf("| n | neighbours | scheme | N = #{p<=0.05} | sd(N) over seeds | sd if independent | ratio | corr adjacent | corr n/2 apart |\n");
    printf("|--:|:--|:--|--:|--:|--:|:--|--:|--:|\n");
    const int sizes[] = {900, 3600, 10000};
    for (int vary = 0; vary < 2; vary++) for (int n : sizes) {
        if (vary && n != 3600) continue;
        for (int s = 0; s < 3; s++) {
            Result a = {0, 0, 0, 0, 0, 0}; double lo = 1e9, hi = 0;
            for (int dset = 0; dset < D; dset++) {
                Result r = measure(make_data(n, vary != 0, 42 + dset), (Scheme)s, P, M);
                a.ratio += r.ratio / D; a.sdN += r.sdN / D; a.sdInd += r.sdInd / D; a.meanN += r.meanN / D;
                a.corr_adj += r.corr_adj / D; a.corr_far += r.corr_far / D;
                lo = std::min(lo, r.ratio); hi = std::max(hi, r.ratio);
            }
            printf("| %d | %s | %s | %.1f | %.2f | %.2f | %.2f [%.2f, %.2f] | %+.3f | %+.3f |\n", n, vary ? "3..8 random" : "4",
                   NAMES[s], a.meanN, a.sdN, a.sdInd, a.ratio, lo, hi, a.corr_adj, a.corr_far);
            fflush(stdout);
        }
    }
    return 0;
}
