#ifdef __APPLE__

// NOTE: compile with -fobjc-arc

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_lisa.h"
#ifndef STANDALONE_TEST // test_metal_lisa.mm provides a wx-free GalElement
#include "../ShapeOperations/GalWeight.h"
#endif
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

static std::mutex s_metal_mutex;

static id<MTLDevice> metal_device()
{
    // Apple Silicon only (M1 is MTLGPUFamilyApple7): Intel Macs fall back to OpenCL
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    // (GEODA_METAL_ANY_GPU: for tests on the paravirtual GPU of CI runners)
    static bool supported = device && ([device supportsFamily:MTLGPUFamilyApple7] || getenv("GEODA_METAL_ANY_GPU"));
    return supported ? device : nil;
}

bool is_metal_supported()
{
    return metal_device() != nil;
}

// Compile (once per kernel and max_nbrs) the kernel from the .metal source
static id<MTLComputePipelineState> metal_pipeline(const char* metal_path, NSString* kernel_name, int max_nbrs)
{
    static std::map<std::pair<std::string, int>, id<MTLComputePipelineState> > cache;
    std::pair<std::string, int> key([kernel_name UTF8String], max_nbrs);
    if (cache.find(key) != cache.end()) return cache[key];

    NSError* error = nil;
    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:metal_path]
                                              encoding:NSUTF8StringEncoding error:&error];
    if (!src) {
        std::cerr << "Metal: Could not read kernel from " << metal_path << "\n";
        return nil;
    }
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.preprocessorMacros = @{ @"MAX_NBRS" : @(max_nbrs) };
    id<MTLLibrary> library = [metal_device() newLibraryWithSource:src options:options error:&error];
    id<MTLFunction> func = library ? [library newFunctionWithName:kernel_name] : nil;
    id<MTLComputePipelineState> pipeline = func ? [metal_device() newComputePipelineStateWithFunction:func error:&error] : nil;
    if (!pipeline) {
        std::cerr << "Metal: Could not build kernel " << [kernel_name UTF8String];
        if (error) std::cerr << ": " << [[error localizedDescription] UTF8String];
        std::cerr << "\n";
        return nil;
    }
    cache[key] = pipeline;
    return pipeline;
}

// 128 bit two's complement fixed point: x * 2^shift truncated to an integer, clamped to +-2^126
struct Fixed128 {
    uint64_t lo;
    uint64_t hi;
};

static Fixed128 to_fixed128(double x, int shift)
{
    double scaled = ldexp(x, shift);
    const double limit = ldexp(1.0, 126);
    if (scaled > limit) scaled = limit;
    if (scaled < -limit) scaled = -limit;
    __int128 q = (__int128)scaled;
    Fixed128 f = { (uint64_t)q, (uint64_t)(q >> 64) };
    return f;
}

// Number of neighbors (excluding self) to permute for each observation, 0 for isolates,
// -1 if the only neighbor is the observation itself
static bool metal_num_nbrs(int rows, GalElement* w, bool skip_isolates, std::vector<int>& num_nbrs, int& max_nbrs)
{
    int candidates = 0;
    max_nbrs = 0;
    num_nbrs.resize(rows);
    for (int i = 0; i < rows; i++) {
        num_nbrs[i] = (int)w[i].Size();
        if (num_nbrs[i] > 0 || !skip_isolates) candidates++;
        if (w[i].Check(i)) num_nbrs[i] -= 1;
        // only neighbor is itself: no permutation test, but still drawn (CPU tests Size() > 0)
        if (num_nbrs[i] == 0 && w[i].Size() > 0) num_nbrs[i] = -1;
        if (num_nbrs[i] > max_nbrs) max_nbrs = num_nbrs[i];
    }
    // not enough observations to draw from: the permutation would never end
    return max_nbrs < candidates;
}

// Run the kernel for each observation. Kernel arguments are n, permutations, seed,
// num_nbrs, inputs... and the output count_larger (< 0 means no permutation test)
typedef std::pair<const void*, size_t> MetalInput; // data, size in bytes

static bool metal_run(const char* metal_path, NSString* kernel_name, int max_nbrs, int rows, int permutations,
                      unsigned long long last_seed_used, const std::vector<int>& num_nbrs,
                      std::vector<MetalInput> inputs, std::vector<int>& count_larger)
{
    @autoreleasepool {
        std::lock_guard<std::mutex> lock(s_metal_mutex);
        id<MTLDevice> device = metal_device();
        if (!device) return false;
        static id<MTLCommandQueue> queue = [device newCommandQueue];

        // size of kernel's hash table of drawn observations: power of 2, load factor
        // at most 0.5 (rounding up also limits recompilations)
        int buf_size = 64;
        while (buf_size < 2 * max_nbrs) buf_size *= 2;
        id<MTLComputePipelineState> pipeline = metal_pipeline(metal_path, kernel_name, buf_size);
        if (!queue || !pipeline) return false;

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&rows length:sizeof(int) atIndex:0];
        [encoder setBytes:&permutations length:sizeof(int) atIndex:1];
        [encoder setBytes:&last_seed_used length:sizeof(unsigned long long) atIndex:2];

        // unified memory shared buffers
        inputs.insert(inputs.begin(), MetalInput(num_nbrs.data(), sizeof(int)*rows));
        NSUInteger index = 3;
        for (size_t i = 0; i < inputs.size(); i++) {
            id<MTLBuffer> buf = [device newBufferWithBytes:inputs[i].first length:inputs[i].second options:MTLResourceStorageModeShared];
            if (!buf) return false;
            [encoder setBuffer:buf offset:0 atIndex:index++];
        }
        id<MTLBuffer> bufCount = [device newBufferWithLength:sizeof(int)*rows options:MTLResourceStorageModeShared];
        if (!bufCount) return false;
        [encoder setBuffer:bufCount offset:0 atIndex:index];

        NSUInteger group_size = pipeline.maxTotalThreadsPerThreadgroup;
        if (group_size > (NSUInteger)rows) group_size = rows;
        [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            std::cerr << "Metal: Command buffer execution failed.\n";
            return false;
        }
        const int* result = (const int*)[bufCount contents];
        count_larger.assign(result, result + rows);
        return true;
    }
}

bool metal_lisa(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                double* values, double* local_moran, GalElement* w, double* p)
{
    if (rows <= 0) return false;

    int max_nbrs = 0;
    std::vector<int> num_nbrs, count_larger;
    if (!metal_num_nbrs(rows, w, true, num_nbrs, max_nbrs)) return false;

    // No fp64 on Apple GPUs: doubles are passed as 128 bit fixed point numbers (exact sums).
    // permuted lag * values[i] >= local_moran[i] (row-standardized weights) is tested in the
    // kernel as sum of permuted neighbors >=(<=) lag_sum[i] for positive (negative) values[i]
    double max_abs = 0;
    for (int i = 0; i < rows; i++) {
        if (!std::isfinite(values[i]) || !std::isfinite(local_moran[i])) return false;
        if (fabs(values[i]) > max_abs) max_abs = fabs(values[i]);
    }
    // values become integers below 2^110: sums of less than 2^17 of them fit
    int exponent = 0;
    frexp(max_abs, &exponent);
    int shift = 110 - exponent;
    std::vector<Fixed128> val(rows), lag_sum(rows);
    std::vector<int> value_sign(rows);
    for (int i = 0; i < rows; i++) {
        val[i] = to_fixed128(values[i], shift);
        value_sign[i] = (values[i] > 0) - (values[i] < 0);
        double s = values[i] != 0 ? local_moran[i] * num_nbrs[i] / values[i] : 0;
        lag_sum[i] = to_fixed128(s, shift);
    }

    std::vector<MetalInput> inputs;
    inputs.push_back(MetalInput(val.data(), sizeof(Fixed128)*rows));
    inputs.push_back(MetalInput(lag_sum.data(), sizeof(Fixed128)*rows));
    inputs.push_back(MetalInput(value_sign.data(), sizeof(int)*rows));
    if (!metal_run(metal_path, @"lisa_metal", max_nbrs, rows, permutations, last_seed_used,
                   num_nbrs, inputs, count_larger)) return false;

    for (int i = 0; i < rows; i++) {
        if (count_larger[i] >= 0) p[i] = (count_larger[i] + 1.0) / (permutations + 1);
    }
    return true;
}

bool metal_localjoincount(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                          int num_vars, int* zz, double* local_jc, GalElement* w, double* p)
{
    if (rows <= 0) return false;

    int max_nbrs = 0;
    std::vector<int> num_nbrs, count_larger;
    if (!metal_num_nbrs(rows, w, false, num_nbrs, max_nbrs)) return false;

    std::vector<int> local_jc_i(local_jc, local_jc + rows);

    std::vector<MetalInput> inputs;
    inputs.push_back(MetalInput(zz, sizeof(int)*rows));
    inputs.push_back(MetalInput(local_jc_i.data(), sizeof(int)*rows));
    if (!metal_run(metal_path, @"localjc_metal", max_nbrs, rows, permutations, last_seed_used,
                   num_nbrs, inputs, count_larger)) return false;

    for (int i = 0; i < rows; i++) {
        if (local_jc[i] == 0) p[i] = 0;
        else if (count_larger[i] >= 0) p[i] = (count_larger[i] + 1.0) / (permutations + 1.0);
    }
    return true;
}

#endif // __APPLE__
