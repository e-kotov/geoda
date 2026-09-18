#ifdef __APPLE__

// NOTE: compile with -fobjc-arc

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_lisa.h"
#ifndef STANDALONE_TEST // test_metal_lisa.mm provides a wx-free GalElement
#include "../ShapeOperations/GalWeight.h"
#endif
#include <climits>
#include <algorithm>
#include <cmath>
#include <cstdio>
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

// Compile (once per kernel and set of macros) the kernel from the .metal source
static id<MTLComputePipelineState> metal_pipeline(const char* metal_path, NSString* kernel_name,
                                                  int max_nbrs, int n_periods, bool has_undef)
{
    static std::map<std::string, id<MTLComputePipelineState> > cache;
    char buf[128];
    snprintf(buf, sizeof(buf), "%s:%d:%d:%d", [kernel_name UTF8String], max_nbrs, n_periods, (int)has_undef);
    std::string key(buf);
    if (cache.find(key) != cache.end()) return cache[key];

    NSError* error = nil;
    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:metal_path]
                                              encoding:NSUTF8StringEncoding error:&error];
    if (!src) {
        std::cerr << "Metal: Could not read kernel from " << metal_path << "\n";
        return nil;
    }
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.preprocessorMacros = @{ @"MAX_NBRS" : @(max_nbrs), @"N_PERIODS" : @(n_periods),
                                    @"HAS_UNDEF" : @(has_undef ? 1 : 0) };
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

static bool fixed128_less(const Fixed128& a, const Fixed128& b)
{
    return a.hi != b.hi ? (int64_t)a.hi < (int64_t)b.hi : a.lo < b.lo;
}

static Fixed128 fixed128_add(const Fixed128& a, const Fixed128& b)
{
    Fixed128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1 : 0);
    return r;
}

// Run the kernel for each observation. Kernel arguments are n, permutations, seed, the
// inputs and the output count_larger, one count per time period and observation
// (< 0 means no permutation test)
typedef std::pair<const void*, size_t> MetalInput; // data, size in bytes

static bool metal_run(const char* metal_path, NSString* kernel_name, int max_nbrs, int n_periods,
                      bool has_undef, int rows, int permutations, unsigned long long last_seed_used,
                      const std::vector<MetalInput>& inputs, std::vector<int>& count_larger)
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
        id<MTLComputePipelineState> pipeline = metal_pipeline(metal_path, kernel_name, buf_size,
                                                              n_periods, has_undef);
        if (!queue || !pipeline) return false;

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBytes:&rows length:sizeof(int) atIndex:0];
        [encoder setBytes:&permutations length:sizeof(int) atIndex:1];
        [encoder setBytes:&last_seed_used length:sizeof(unsigned long long) atIndex:2];

        // unified memory shared buffers
        NSUInteger index = 3;
        for (size_t i = 0; i < inputs.size(); i++) {
            id<MTLBuffer> buf = [device newBufferWithBytes:inputs[i].first length:inputs[i].second options:MTLResourceStorageModeShared];
            if (!buf) return false;
            [encoder setBuffer:buf offset:0 atIndex:index++];
        }
        id<MTLBuffer> bufCount = [device newBufferWithLength:sizeof(int)*rows*n_periods options:MTLResourceStorageModeShared];
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
        count_larger.assign(result, result + rows * n_periods);
        return true;
    }
}

// Neighbors to draw for each observation and the observations the draw may pick, exactly
// as AbstractCoordinator::CalcPseudoP_range(): the largest neighbor count over the time
// periods (including its upstream quirks) minus a self-neighbor, and Size() > 0 in the
// LAST period, the one its rejection rule reads.
// Returns false when the CPU's draw could not terminate or when the sums could overflow.
static bool metal_lisa_nbrs(const GdaLisaPerm& perm, std::vector<int>& num_nbrs,
                            std::vector<unsigned char>& draw_ok, int& max_nbrs)
{
    const int rows = perm.rows, tms = perm.num_time_vals;
    num_nbrs.assign(rows, 0);
    draw_ok.assign(rows, 0);
    max_nbrs = 0;
    int candidates = 0;
    for (int i = 0; i < rows; i++) {
        for (int t = 0; t < tms; t++) {
            GalElement* w = perm.w[t];
            if (w[i].Size() > num_nbrs[i]) {
                num_nbrs[i] = (int)w[i].Size();
                if (w[i].Check(i)) num_nbrs[i] -= 1; // exclude self from neighbors
            }
        }
        draw_ok[i] = perm.w[tms - 1][i].Size() > 0;
        candidates += draw_ok[i];
        if (num_nbrs[i] > max_nbrs) max_nbrs = num_nbrs[i];
    }
    // not enough observations to draw from: the permutation would never end.
    // Fixed point values are below 2^110, so sumAbs + |lag_sum| < 2 k 2^110 = k 2^111
    // stays below 2^127 exactly while k < 2^16: the bound is tight, not conservative.
    return max_nbrs < candidates && max_nbrs < (1 << 16);
}

// The observed median times two, as the sum of the fixed point values of the one or two
// middle neighbors, so that the kernel can compare it with the permuted median exactly
// (LisaCoordinator::Calc():508-521 takes GenUtils::Median() of the non-self neighbors)
static Fixed128 metal_median2(const GalElement& w, int i, const Fixed128* values,
                              std::vector<Fixed128>& nbr_data)
{
    nbr_data.clear();
    for (long j = 0, sz = w.Size(); j < sz; j++) {
        if (w[j] != i) nbr_data.push_back(values[w[j]]);
    }
    Fixed128 zero = {0, 0};
    if (nbr_data.empty()) return zero;
    const size_t lo_rank = (nbr_data.size() - 1) / 2, hi_rank = nbr_data.size() / 2;
    std::nth_element(nbr_data.begin(), nbr_data.begin() + hi_rank, nbr_data.end(), fixed128_less);
    Fixed128 hi = nbr_data[hi_rank];
    std::nth_element(nbr_data.begin(), nbr_data.begin() + lo_rank, nbr_data.begin() + hi_rank + 1,
                     fixed128_less);
    return fixed128_add(nbr_data[lo_rank], hi);
}

bool metal_lisa(const char* metal_path, const GdaLisaPerm& perm, const std::vector<double*>& p)
{
    const int rows = perm.rows, tms = perm.num_time_vals;
    // the kernel indexes the time periods as t * n + i in 32 bit integers
    if (rows > 0 && tms > 0 && (long long)rows * tms > INT_MAX) return false;
    if (rows <= 0 || tms <= 0 || (int)perm.data1.size() < tms || (int)perm.lagged.size() < tms ||
        (int)perm.lags.size() < tms || (int)perm.w.size() < tms || (int)p.size() < tms ||
        (perm.undef && (int)perm.undef->size() < tms)) {
        return false;
    }

    int max_nbrs = 0;
    std::vector<int> num_nbrs, count_larger;
    std::vector<unsigned char> draw_ok;
    if (!metal_lisa_nbrs(perm, num_nbrs, draw_ok, max_nbrs)) return false;

    // Only undefined observations that can be drawn change anything: they are left out of
    // the permuted lag, which makes the number of valid neighbors vary from permutation to
    // permutation. Isolates are undefined too, but are never drawn.
    bool has_undef = false;
    for (int t = 0; perm.undef && t < tms && !has_undef; t++) {
        for (int i = 0; i < rows; i++) {
            if ((*perm.undef)[t][i] && draw_ok[i]) { has_undef = true; break; }
        }
    }

    // No fp64 on Apple GPUs: doubles are passed as 128 bit fixed point numbers (exact
    // sums), one array per time period. value_sign is the sign of data1[i], which decides
    // the direction of the comparison, and count_empty its answer when every drawn
    // neighbor is undefined.
    const size_t sz = (size_t)rows * tms;
    std::vector<Fixed128> val(sz), lag(sz);
    std::vector<int> value_sign(sz);
    std::vector<unsigned char> undef(has_undef ? sz : 1, 0), count_empty(has_undef ? sz : 1, 0);
    std::vector<Fixed128> nbr_data;

    for (int t = 0; t < tms; t++) {
        const double* data1 = perm.data1[t];
        const double* lagged = perm.lagged[t];
        const double* lags = perm.lags[t];
        double max_abs = 0;
        for (int i = 0; i < rows; i++) {
            // data1[i] decides the direction of the comparison of EVERY observation,
            // including the undefined ones (StandardizeData() leaves them a value), so a
            // non-finite value there is refused too
            if (!std::isfinite(data1[i])) return false;
            if (perm.undef && (*perm.undef)[t][i]) continue; // undefined values are not lagged
            if (!std::isfinite(lagged[i]) || !std::isfinite(lags[i])) return false;
            if (fabs(lagged[i]) > max_abs) max_abs = fabs(lagged[i]);
            if (fabs(lags[i]) > max_abs) max_abs = fabs(lags[i]);
        }
        // values become integers below 2^110 (the scale is a power of two: exact)
        int exponent = 0;
        frexp(max_abs, &exponent);
        const int shift = 110 - exponent;
        Fixed128* val_t = &val[(size_t)t * rows];
        for (int i = 0; i < rows; i++) {
            const bool is_undef = perm.undef && (*perm.undef)[t][i];
            const size_t k = (size_t)t * rows + i;
            val_t[i] = to_fixed128(is_undef ? 0 : lagged[i], shift);
            value_sign[k] = (data1[i] > 0) - (data1[i] < 0);
            if (!perm.using_median) {
                lag[k] = to_fixed128(is_undef ? 0 : lags[i], shift); // lags[i] is 0 if undefined
            }
            if (has_undef) {
                undef[k] = is_undef;
                // local_moran[i] is data1[i] * lags[i], and 0 for undefined observations
                count_empty[k] = is_undef || data1[i] * lags[i] <= 0;
            }
        }
        if (perm.using_median) {
            for (int i = 0; i < rows; i++) {
                const bool is_undef = perm.undef && (*perm.undef)[t][i];
                Fixed128 zero = {0, 0};   // Calc() leaves lags_vecs[t][i] at 0 for those
                lag[(size_t)t * rows + i] = is_undef ? zero
                                          : metal_median2(perm.w[t][i], i, val_t, nbr_data);
            }
        }
    }

    std::vector<MetalInput> inputs;
    inputs.push_back(MetalInput(num_nbrs.data(), sizeof(int)*rows));
    inputs.push_back(MetalInput(draw_ok.data(), draw_ok.size()));
    inputs.push_back(MetalInput(undef.data(), undef.size()));
    inputs.push_back(MetalInput(count_empty.data(), count_empty.size()));
    inputs.push_back(MetalInput(value_sign.data(), sizeof(int)*sz));
    inputs.push_back(MetalInput(val.data(), sizeof(Fixed128)*sz));
    inputs.push_back(MetalInput(lag.data(), sizeof(Fixed128)*sz));
    if (!metal_run(metal_path, perm.using_median ? @"lisa_median_metal" : @"lisa_metal",
                   max_nbrs, tms, has_undef, rows, perm.permutations, perm.last_seed_used,
                   inputs, count_larger)) return false;

    for (int t = 0; t < tms; t++) {
        for (int i = 0; i < rows; i++) {
            int c = count_larger[(size_t)t * rows + i];
            if (c >= 0) p[t][i] = (c + 1.0) / (perm.permutations + 1);
        }
    }
    return true;
}

bool metal_lisa(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                double* values, double* local_moran, GalElement* w, double* p)
{
    if (rows <= 0) return false;

    // the observed lag, which LisaCoordinator::Calc() keeps in lags_vecs
    std::vector<double> lags(rows, 0);
    for (int i = 0; i < rows; i++) {
        if (!std::isfinite(values[i]) || !std::isfinite(local_moran[i])) return false;
        if (values[i] != 0) lags[i] = local_moran[i] / values[i];
    }

    GdaLisaPerm perm;
    perm.rows = rows;
    perm.permutations = permutations;
    perm.num_time_vals = 1;
    perm.last_seed_used = last_seed_used;
    perm.data1.push_back(values);
    perm.lagged.push_back(values);
    perm.lags.push_back(lags.data());
    perm.w.push_back(w);
    return metal_lisa(metal_path, perm, std::vector<double*>(1, p));
}

bool metal_localjoincount(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                          int num_vars, int* zz, double* local_jc, GalElement* w, double* p,
                          const std::vector<bool>* undef)
{
    if (rows <= 0) return false;

    // Neighbors to permute (excluding a self-neighbor) and the observations the draw may
    // pick: JCCoordinator::CalcPseudoP_range() rejects undefined observations, not
    // neighborless ones, and computes nothing for an undefined observation.
    int max_nbrs = 0, candidates = 0;
    std::vector<int> num_nbrs(rows), count_larger;
    std::vector<unsigned char> draw_ok(rows);
    for (int i = 0; i < rows; i++) {
        num_nbrs[i] = (int)w[i].Size();
        if (w[i].Check(i)) num_nbrs[i] -= 1;
        draw_ok[i] = undef ? !(*undef)[i] : 1;
        candidates += draw_ok[i];
        // observations the CPU skips (:623, :625) are not permuted by the kernel either
        if (draw_ok[i] && local_jc[i] != 0 && num_nbrs[i] > max_nbrs) max_nbrs = num_nbrs[i];
    }
    // not enough observations to draw from: the permutation would never end
    if (max_nbrs >= candidates) return false;

    std::vector<int> local_jc_i(local_jc, local_jc + rows);

    std::vector<MetalInput> inputs;
    inputs.push_back(MetalInput(num_nbrs.data(), sizeof(int)*rows));
    inputs.push_back(MetalInput(draw_ok.data(), draw_ok.size()));
    inputs.push_back(MetalInput(zz, sizeof(int)*rows));
    inputs.push_back(MetalInput(local_jc_i.data(), sizeof(int)*rows));
    if (!metal_run(metal_path, @"localjc_metal", max_nbrs, 1, undef != 0, rows, permutations,
                   last_seed_used, inputs, count_larger)) return false;

    for (int i = 0; i < rows; i++) {
        if (undef && (*undef)[i]) continue; // no pseudo p-value for undefined observations
        if (local_jc[i] == 0) p[i] = 0;
        else if (count_larger[i] >= 0) p[i] = (count_larger[i] + 1.0) / (permutations + 1.0);
    }
    return true;
}

#endif // __APPLE__
