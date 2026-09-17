#import <Foundation/Foundation.h>
#define STANDALONE_TEST 1
#include "metal_lisa.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << "Testing Apple Metal LISA GPU Permutations\n";
    std::cout << "========================================\n";

    if (!is_metal_supported()) {
        std::cerr << "Metal is not supported on this device/environment.\n";
        return 1;
    }
    std::cout << "[PASS] Apple Metal device detected.\n";

    const char* lisa_metal_path = (argc > 1) ? argv[1] : "Algorithms/lisa_kernel.metal";
    const char* localjc_metal_path = (argc > 2) ? argv[2] : "Algorithms/localjc_kernel.metal";

    int rows = 300;
    int permutations = 999;
    unsigned long long seed = 123456789ULL;

    std::vector<double> values(rows);
    std::vector<double> local_moran(rows);
    std::vector<double> p_metal(rows, 0.0);
    std::vector<GalElement> w(rows);

    // Create synthetic ring topology (each node connected to 4 neighbors)
    for (int i = 0; i < rows; i++) {
        values[i] = std::sin((double)i * 0.1);
        w[i].SetSizeNbrs(4);
        w[i].SetNbr(0, (i + 1) % rows);
        w[i].SetNbr(1, (i + 2) % rows);
        w[i].SetNbr(2, (i + rows - 1) % rows);
        w[i].SetNbr(3, (i + rows - 2) % rows);
    }

    // Compute observed Local Moran
    for (int i = 0; i < rows; i++) {
        double lag = 0.0;
        for (long j = 0; j < w[i].Size(); j++) {
            lag += values[w[i][j]];
        }
        lag /= (double)w[i].Size();
        local_moran[i] = lag * values[i];
    }

    std::cout << "Running metal_lisa with " << rows << " rows and " << permutations << " permutations...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = metal_lisa(lisa_metal_path, rows, permutations, seed,
                              values.data(), local_moran.data(), w.data(), p_metal.data());
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    if (!success) {
        std::cerr << "[FAIL] metal_lisa failed execution!\n";
        return 1;
    }
    std::cout << "[PASS] metal_lisa executed successfully in " << duration.count() << " ms.\n";

    // Validate p-values
    int valid_count = 0;
    for (int i = 0; i < rows; i++) {
        if (p_metal[i] > 0.0 && p_metal[i] <= 1.0) {
            valid_count++;
        } else {
            std::cerr << "[FAIL] Invalid p-value at index " << i << ": " << p_metal[i] << "\n";
            return 1;
        }
    }
    std::cout << "[PASS] All " << valid_count << " p-values in valid range (0, 1]. Sample p[0]="
              << p_metal[0] << ", p[1]=" << p_metal[1] << ", p[2]=" << p_metal[2] << "\n";

    // Test Local Join Count
    std::cout << "\nTesting metal_localjoincount...\n";
    std::vector<int> zz(rows);
    std::vector<double> local_jc(rows);
    std::vector<double> p_jc(rows, 0.0);
    for (int i = 0; i < rows; i++) {
        zz[i] = (i % 2 == 0) ? 1 : 0;
    }
    for (int i = 0; i < rows; i++) {
        double lag = 0.0;
        for (long j = 0; j < w[i].Size(); j++) {
            lag += zz[w[i][j]];
        }
        local_jc[i] = lag;
    }

    start_time = std::chrono::high_resolution_clock::now();
    bool jc_success = metal_localjoincount(localjc_metal_path, rows, permutations, seed,
                                           1, zz.data(), local_jc.data(), w.data(), p_jc.data());
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time - start_time;

    if (!jc_success) {
        std::cerr << "[FAIL] metal_localjoincount failed execution!\n";
        return 1;
    }
    std::cout << "[PASS] metal_localjoincount executed successfully in " << duration.count() << " ms.\n";

    for (int i = 0; i < rows; i++) {
        if (p_jc[i] < 0.0 || p_jc[i] > 1.0) {
            std::cerr << "[FAIL] Invalid Join Count p-value at index " << i << ": " << p_jc[i] << "\n";
            return 1;
        }
    }
    std::cout << "[PASS] All Join Count p-values valid. Sample p_jc[0]=" << p_jc[0] << ", p_jc[1]=" << p_jc[1] << "\n";

    std::cout << "\n========================================\n";
    std::cout << "ALL APPLE METAL TESTS PASSED!\n";
    std::cout << "========================================\n";
    return 0;
}
