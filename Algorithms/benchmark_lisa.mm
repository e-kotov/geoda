#import <Foundation/Foundation.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <iomanip>
#include <algorithm>

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
#include "test_data_natregimes.h" // GeoDa sample data, see make_test_data.R

// The 64 bit integer hash of Gda::ThomasWangHashDouble(), before it is scaled to a double
inline unsigned long long ThomasWangHash(unsigned long long key)
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

// Gda::ThomasWangHashDouble()
inline double ThomasWangHashDouble(unsigned long long key)
{
    return 5.42101086242752217E-20 * ThomasWangHash(key);
}

// Reference CPU implementation for a chunk of observations
void cpu_lisa_worker(int start_row, int end_row, int n, int permutations,
                     unsigned long long last_seed, const double* values,
                     const double* local_moran, const GalElement* w, double* p)
{
    std::vector<int> rnd_numbers(256);

    for (int i = start_row; i < end_row; i++) {
        int numNeighbors = (int)w[i].Size();
        if (numNeighbors == 0) {
            p[i] = 1.0;
            continue;
        }

        // the keys of the GPU kernels: every permutation has its own sequence (GPU-7)
        unsigned long long obs_key = ThomasWangHash((unsigned long long)i + last_seed);
        double max_rand = (double)(n - 1);
        int countLarger = 0;

        if ((int)rnd_numbers.size() < numNeighbors * 2) {
            rnd_numbers.resize(numNeighbors * 2);
        }

        for (int perm = 0; perm < permutations; perm++) {
            unsigned long long seed_start = ThomasWangHash(obs_key + (unsigned long long)perm);
            int rand_cnt = 0;
            double permutedLag = 0.0;

            while (rand_cnt < numNeighbors) {
                double rng_val = ThomasWangHashDouble(seed_start++) * max_rand;
                int newRandom = (int)floor(rng_val + 0.5);

                if (newRandom != i) {
                    bool is_valid = true;
                    for (int j = 0; j < rand_cnt; j++) {
                        if (rnd_numbers[j] == newRandom) {
                            is_valid = false;
                            break;
                        }
                    }
                    if (is_valid) {
                        permutedLag += values[newRandom];
                        rnd_numbers[rand_cnt++] = newRandom;
                    }
                }
            }

            permutedLag /= numNeighbors;
            double localMoranPermuted = permutedLag * values[i];
            if (localMoranPermuted > local_moran[i]) {
                countLarger++;
            }
        }

        if (permutations - countLarger <= countLarger) {
            countLarger = permutations - countLarger;
        }

        p[i] = (double)(countLarger + 1) / (double)(permutations + 1);
    }
}

// CPU Multi-threaded execution
void cpu_lisa_multithreaded(int n, int permutations, unsigned long long seed,
                            const double* values, const double* local_moran,
                            const GalElement* w, double* p, int num_threads)
{
    std::vector<std::thread> threads;
    int chunk_size = (n + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, n);
        if (start < end) {
            threads.emplace_back(cpu_lisa_worker, start, end, n, permutations,
                                 seed, values, local_moran, w, p);
        }
    }

    for (auto& th : threads) {
        th.join();
    }
}

struct BenchmarkResult {
    int n;
    int permutations;
    double cpu_single_ms;
    double cpu_multi_ms;
    double metal_gpu_ms;
    double speedup_vs_multi;
    double speedup_vs_single;
    double max_pval_diff;
};

BenchmarkResult run_benchmark(int n, int k, int permutations, const char* metal_kernel_path, int num_threads)
{
    unsigned long long seed = 42ULL;
    std::vector<double> values(n);
    std::vector<double> local_moran(n);
    std::vector<double> p_cpu_single(n, 0.0);
    std::vector<double> p_cpu_multi(n, 0.0);
    std::vector<double> p_metal(n, 0.0);
    std::vector<GalElement> w(n);

    // Real data (GeoDa's "US Homicides" sample, hr90, queen weights) if n matches
    bool real_data = (n == natregimes_n);
    for (int i = 0; real_data && i < n; i++) {
        values[i] = natregimes_x[i];
        int nn = natregimes_nbr_offset[i + 1] - natregimes_nbr_offset[i];
        w[i].SetSizeNbrs(nn);
        for (int j = 0; j < nn; j++) w[i].SetNbr(j, natregimes_nbrs[natregimes_nbr_offset[i] + j]);
    }

    // otherwise generate synthetic spatial grid / ring data
    for (int i = 0; !real_data && i < n; i++) {
        values[i] = std::sin((double)i * 0.05) + std::cos((double)i * 0.02);
        w[i].SetSizeNbrs(k);
        for (int j = 0; j < k; j++) {
            int offset = (j % 2 == 0) ? (j / 2 + 1) : -(j / 2 + 1);
            int nbr = (i + offset + n) % n;
            w[i].SetNbr(j, nbr);
        }
    }

    // Compute observed Local Moran
    for (int i = 0; i < n; i++) {
        double lag = 0.0;
        for (long j = 0; j < w[i].Size(); j++) {
            lag += values[w[i][j]];
        }
        lag /= (double)w[i].Size();
        local_moran[i] = lag * values[i];
    }

    // 1. CPU Multi-threaded
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_lisa_multithreaded(n, permutations, seed, values.data(), local_moran.data(), w.data(), p_cpu_multi.data(), num_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_multi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. CPU Single-threaded (only if n <= 5000 to save time)
    double cpu_single_ms = 0.0;
    if (n <= 5000) {
        t0 = std::chrono::high_resolution_clock::now();
        cpu_lisa_worker(0, n, n, permutations, seed, values.data(), local_moran.data(), w.data(), p_cpu_single.data());
        t1 = std::chrono::high_resolution_clock::now();
        cpu_single_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    } else {
        cpu_single_ms = cpu_multi_ms * num_threads * 0.85; // estimated
    }

    // 3. Apple Metal GPU
    t0 = std::chrono::high_resolution_clock::now();
    bool ok = metal_lisa(metal_kernel_path, n, permutations, seed,
                         values.data(), local_moran.data(), w.data(), p_metal.data());
    t1 = std::chrono::high_resolution_clock::now();
    double metal_gpu_ms = ok ? std::chrono::duration<double, std::milli>(t1 - t0).count() : -1.0;

    // Numerical validation (compare p-values between CPU multi and GPU)
    double max_diff = 0.0;
    if (ok) {
        for (int i = 0; i < n; i++) {
            double diff = std::fabs(p_cpu_multi[i] - p_metal[i]);
            if (diff > max_diff) max_diff = diff;
        }
    }

    BenchmarkResult res;
    res.n = n;
    res.permutations = permutations;
    res.cpu_single_ms = cpu_single_ms;
    res.cpu_multi_ms = cpu_multi_ms;
    res.metal_gpu_ms = metal_gpu_ms;
    res.speedup_vs_multi = (metal_gpu_ms > 0) ? (cpu_multi_ms / metal_gpu_ms) : 0.0;
    res.speedup_vs_single = (metal_gpu_ms > 0) ? (cpu_single_ms / metal_gpu_ms) : 0.0;
    res.max_pval_diff = max_diff;
    return res;
}

int main(int argc, char* argv[])
{
    const char* kernel_path = (argc > 1) ? argv[1] : "Algorithms/lisa_kernel.metal";
    int num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads < 1) num_threads = 4;

    std::cout << "==================================================================\n";
    std::cout << "   GeoDa Apple Metal vs CPU Permutation Performance Benchmark     \n";
    std::cout << "==================================================================\n";
    std::cout << "Hardware Threads: " << num_threads << "\n";
    std::cout << "Metal Supported:  " << (is_metal_supported() ? "YES" : "NO") << "\n";
    std::cout << "Kernel Path:      " << kernel_path << "\n\n";

    if (!is_metal_supported()) {
        std::cerr << "Error: Apple Metal GPU is not available on this system.\n";
        return 1;
    }

    struct BenchmarkConfig {
        std::string name;
        int n;
        int k;
        int permutations;
    };

    std::vector<BenchmarkConfig> configs = {
        {"US Homicides hr90 (real data)", natregimes_n, 6, 99999},
        {"50k Grid", 50176, 8, 999},
        {"All Denmark", 179674, 8, 999}
    };

    std::vector<BenchmarkResult> results;

    for (const auto& cfg : configs) {
        double total_m = (double)cfg.n * (double)cfg.permutations / 1e6;
        std::cout << ">>> Running " << cfg.name << " (N = " << cfg.n << ", P = " << cfg.permutations
                  << ", Total = " << std::fixed << std::setprecision(2) << total_m << "M perms, k = " << cfg.k << ")...\n";
        BenchmarkResult r = run_benchmark(cfg.n, cfg.k, cfg.permutations, kernel_path, num_threads);
        results.push_back(r);
        std::cout << "  -> CPU Multi-Core (" << num_threads << " threads): " << std::fixed << std::setprecision(3) << (r.cpu_multi_ms / 1000.0) << " s\n";
        std::cout << "  -> Apple Metal GPU:               " << std::fixed << std::setprecision(3) << (r.metal_gpu_ms / 1000.0) << " s\n";
        std::cout << "  -> GPU Speedup:                   " << std::fixed << std::setprecision(2) << r.speedup_vs_multi << "x faster\n";
        std::cout << "  -> Max p-val diff:                " << std::setprecision(4) << r.max_pval_diff << "\n\n";
    }

    // Print summary table in Markdown format
    std::stringstream md;
    md << "### 🚀 GeoDa Permutation Performance: CPU vs Apple Metal GPU\n\n";
    md << "| Dataset | $N$ (Cells) | $P$ (Perms) | Total Permutations | CPU (" << num_threads << " Cores) | Apple Metal GPU | Speedup (vs Multi-Core) | Max p-value Diff |\n";
    md << "|:---|:---|:---|:---|:---|:---|:---|:---|\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        const auto& cfg = configs[i];
        double total_m = (double)cfg.n * (double)cfg.permutations / 1e6;
        md << "| **" << cfg.name << "** | " << r.n << " | " << r.permutations << " | "
           << std::fixed << std::setprecision(2) << total_m << "M | "
           << std::fixed << std::setprecision(3) << (r.cpu_multi_ms / 1000.0) << " s | **"
           << std::fixed << std::setprecision(3) << (r.metal_gpu_ms / 1000.0) << " s** | **"
           << std::fixed << std::setprecision(1) << r.speedup_vs_multi << "x** | "
           << std::setprecision(4) << r.max_pval_diff << " |\n";
    }

    std::cout << md.str() << "\n";

    // Write to GITHUB_STEP_SUMMARY if present
    const char* summary_file = std::getenv("GITHUB_STEP_SUMMARY");
    if (summary_file) {
        std::ofstream sf(summary_file, std::ios::app);
        if (sf.is_open()) {
            sf << md.str();
        }
    }

    // Write CSV
    std::ofstream csv("benchmark-results.csv");
    if (csv.is_open()) {
        csv << "name,n,permutations,cpu_single_ms,cpu_multi_ms,metal_gpu_ms,speedup_vs_multi,speedup_vs_single,max_pval_diff\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            const auto& cfg = configs[i];
            csv << cfg.name << "," << r.n << "," << r.permutations << "," << r.cpu_single_ms << ","
                << r.cpu_multi_ms << "," << r.metal_gpu_ms << ","
                << r.speedup_vs_multi << "," << r.speedup_vs_single << ","
                << r.max_pval_diff << "\n";
        }
    }

    return 0;
}
