#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_lisa.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

bool is_metal_supported()
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return (device != nil);
    }
}

bool metal_lisa(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                double* values, double* local_moran, GalElement* w, double* p)
{
    if (rows <= 0) return false;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal: No default device found.\n";
            return false;
        }

        int max_n_nbrs = 0;
        std::vector<int> num_nbrs(rows, 0);
        int total_nbrs = 0;

        for (int i = 0; i < rows; i++) {
            long nnbrs = w[i].Size();
            if (nnbrs > max_n_nbrs) {
                max_n_nbrs = (int)nnbrs;
            }
            num_nbrs[i] = (int)nnbrs;
            total_nbrs += (int)nnbrs;
        }

        std::vector<int> nbr_idx(total_nbrs > 0 ? total_nbrs : 1, 0);
        size_t idx = 0;
        for (int i = 0; i < rows; i++) {
            long nnbrs = w[i].Size();
            for (long j = 0; j < nnbrs; j++) {
                if (idx < (size_t)total_nbrs) {
                    nbr_idx[idx++] = w[i][j];
                }
            }
        }

        // Read shader source
        std::ifstream t(metal_path);
        std::stringstream buffer;
        buffer << t.rdbuf();
        std::string src_code = buffer.str();
        if (src_code.empty()) {
            std::cerr << "Metal: Could not read kernel from " << metal_path << "\n";
            return false;
        }

        // Replace buffer size marker '123' with max_n_nbrs * 2 (at least 64)
        int buf_size = (max_n_nbrs * 2 < 64) ? 64 : (max_n_nbrs * 2);
        std::string target = "123";
        std::string replacement = std::to_string(buf_size);
        size_t pos = 0;
        while ((pos = src_code.find(target, pos)) != std::string::npos) {
            src_code.replace(pos, target.length(), replacement);
            pos += replacement.length();
        }

        NSError *error = nil;
        NSString *sourceStr = [NSString stringWithUTF8String:src_code.c_str()];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:sourceStr options:options error:&error];
        if (!library) {
            std::cerr << "Metal compile error: " << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }

        id<MTLFunction> kernelFunc = [library newFunctionWithName:@"lisa_metal"];
        if (!kernelFunc) {
            std::cerr << "Metal: Kernel function 'lisa_metal' not found.\n";
            return false;
        }

        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunc error:&error];
        if (!pipelineState) {
            std::cerr << "Metal pipeline state error: " << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) return false;

        // Convert double arrays to float for Apple Silicon GPU
        std::vector<float> values_f(rows);
        std::vector<float> local_moran_f(rows);
        for (int i = 0; i < rows; i++) {
            values_f[i] = (float)values[i];
            local_moran_f[i] = (float)local_moran[i];
        }

        // Unified memory shared buffers
        id<MTLBuffer> bufValues = [device newBufferWithBytes:values_f.data() length:sizeof(float)*rows options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufMoran = [device newBufferWithBytes:local_moran_f.data() length:sizeof(float)*rows options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNumNbrs = [device newBufferWithBytes:num_nbrs.data() length:sizeof(int)*rows options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNbrIdx = [device newBufferWithBytes:nbr_idx.data() length:sizeof(int)*nbr_idx.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufP = [device newBufferWithLength:sizeof(float)*rows options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:pipelineState];
        [encoder setBytes:&rows length:sizeof(int) atIndex:0];
        [encoder setBytes:&permutations length:sizeof(int) atIndex:1];
        [encoder setBytes:&last_seed_used length:sizeof(unsigned long long) atIndex:2];
        [encoder setBuffer:bufValues offset:0 atIndex:3];
        [encoder setBuffer:bufMoran offset:0 atIndex:4];
        [encoder setBuffer:bufNumNbrs offset:0 atIndex:5];
        [encoder setBuffer:bufNbrIdx offset:0 atIndex:6];
        [encoder setBuffer:bufP offset:0 atIndex:7];

        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        NSUInteger maxThreads = pipelineState.maxTotalThreadsPerThreadgroup;
        NSUInteger threadGroupSize = (maxThreads > (NSUInteger)rows) ? (NSUInteger)rows : maxThreads;
        if (threadGroupSize == 0) threadGroupSize = 1;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            std::cerr << "Metal: Command buffer execution failed.\n";
            return false;
        }

        float *p_res = (float *)[bufP contents];
        for (int i = 0; i < rows; i++) {
            p[i] = (double)p_res[i];
        }

        return true;
    }
}

bool metal_localjoincount(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                          int num_vars, int* zz, double* local_jc, GalElement* w, double* p)
{
    if (rows <= 0) return false;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal: No default device found.\n";
            return false;
        }

        int max_n_nbrs = 0;
        std::vector<unsigned short> num_nbrs(rows, 0);
        int total_nbrs = 0;

        for (int i = 0; i < rows; i++) {
            long nnbrs = w[i].Size();
            if (nnbrs > max_n_nbrs) {
                max_n_nbrs = (int)nnbrs;
            }
            num_nbrs[i] = (unsigned short)nnbrs;
            total_nbrs += (int)nnbrs;
        }

        std::vector<unsigned short> nbr_idx(total_nbrs > 0 ? total_nbrs : 1, 0);
        size_t idx = 0;
        for (int i = 0; i < rows; i++) {
            long nnbrs = w[i].Size();
            for (long j = 0; j < nnbrs; j++) {
                if (idx < (size_t)total_nbrs) {
                    nbr_idx[idx++] = (unsigned short)w[i][j];
                }
            }
        }

        std::vector<unsigned short> zz_us(rows);
        std::vector<unsigned short> local_jc_us(rows);
        for (int i = 0; i < rows; i++) {
            zz_us[i] = (unsigned short)zz[i];
            local_jc_us[i] = (unsigned short)local_jc[i];
        }

        // Read shader source
        std::ifstream t(metal_path);
        std::stringstream buffer;
        buffer << t.rdbuf();
        std::string src_code = buffer.str();
        if (src_code.empty()) {
            std::cerr << "Metal: Could not read kernel from " << metal_path << "\n";
            return false;
        }

        int buf_size = (max_n_nbrs * 2 < 64) ? 64 : (max_n_nbrs * 2);
        std::string target = "123";
        std::string replacement = std::to_string(buf_size);
        size_t pos = 0;
        while ((pos = src_code.find(target, pos)) != std::string::npos) {
            src_code.replace(pos, target.length(), replacement);
            pos += replacement.length();
        }

        NSError *error = nil;
        NSString *sourceStr = [NSString stringWithUTF8String:src_code.c_str()];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:sourceStr options:options error:&error];
        if (!library) {
            std::cerr << "Metal compile error: " << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }

        id<MTLFunction> kernelFunc = [library newFunctionWithName:@"localjc_metal"];
        if (!kernelFunc) {
            std::cerr << "Metal: Kernel function 'localjc_metal' not found.\n";
            return false;
        }

        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunc error:&error];
        if (!pipelineState) {
            std::cerr << "Metal pipeline state error: " << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) return false;

        unsigned long u_num_vars = (unsigned long)num_vars;

        id<MTLBuffer> bufZZ = [device newBufferWithBytes:zz_us.data() length:sizeof(unsigned short)*rows options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufJC = [device newBufferWithBytes:local_jc_us.data() length:sizeof(unsigned short)*rows options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNumNbrs = [device newBufferWithBytes:num_nbrs.data() length:sizeof(unsigned short)*rows options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNbrIdx = [device newBufferWithBytes:nbr_idx.data() length:sizeof(unsigned short)*nbr_idx.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufP = [device newBufferWithLength:sizeof(float)*rows options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:pipelineState];
        [encoder setBytes:&rows length:sizeof(int) atIndex:0];
        [encoder setBytes:&permutations length:sizeof(int) atIndex:1];
        [encoder setBytes:&last_seed_used length:sizeof(unsigned long long) atIndex:2];
        [encoder setBytes:&u_num_vars length:sizeof(unsigned long) atIndex:3];
        [encoder setBuffer:bufZZ offset:0 atIndex:4];
        [encoder setBuffer:bufJC offset:0 atIndex:5];
        [encoder setBuffer:bufNumNbrs offset:0 atIndex:6];
        [encoder setBuffer:bufNbrIdx offset:0 atIndex:7];
        [encoder setBuffer:bufP offset:0 atIndex:8];

        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        NSUInteger maxThreads = pipelineState.maxTotalThreadsPerThreadgroup;
        NSUInteger threadGroupSize = (maxThreads > (NSUInteger)rows) ? (NSUInteger)rows : maxThreads;
        if (threadGroupSize == 0) threadGroupSize = 1;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status != MTLCommandBufferStatusCompleted) {
            std::cerr << "Metal: Command buffer execution failed.\n";
            return false;
        }

        float *p_res = (float *)[bufP contents];
        for (int i = 0; i < rows; i++) {
            p[i] = (double)p_res[i];
        }

        return true;
    }
}

#endif // __APPLE__
