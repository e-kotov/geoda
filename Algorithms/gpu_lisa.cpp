#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include "../ShapeOperations/GalWeight.h"
#ifdef __linux__
// do nothing; we got opencl sdk issue on centos
bool gpu_lisa(const char* cl_path, int rows, int permutations, unsigned long long last_seed_used, double* values, double* local_moran, GalElement* w, double* p)
{
    return false;
}

bool gpu_localjoincount(const char* cl_path, int rows, int permutations, unsigned long long last_seed_used, int num_vars, int* zz, double* local_jc, GalElement* w, double* p)
{
    return false;
}
#else

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

#include "../ShapeOperations/GalWeight.h"

using namespace std;

// Number of neighbors to permute for each observation, self excluded, as in
// CalcPseudoP_range(); -1 marks an observation that has only itself as neighbor: no
// permutation test for it, but it is still drawn into permutations of other
// observations (the Local Moran code tests w[newRandom].Size() > 0).
// Returns false if there are not enough observations to draw from, in which case the
// permutation would never end.
static bool prepare_num_nbrs(int rows, GalElement* w, bool skip_isolates, int* num_nbrs, int& max_n_nbrs)
{
    int candidates = 0;
    max_n_nbrs = 0;
    for (int i=0; i<rows; i++) {
        int nnbrs = (int)w[i].Size();
        if (nnbrs > 0 || !skip_isolates) candidates++;
        if (w[i].Check(i)) {
            nnbrs -= 1; // exclude self from neighbors
        }
        num_nbrs[i] = (skip_isolates && nnbrs == 0 && w[i].Size() > 0) ? -1 : nnbrs;
        if (nnbrs > max_n_nbrs) max_n_nbrs = nnbrs;
    }
    return max_n_nbrs < candidates;
}

// Size of the kernel's array of drawn observations: the largest number of neighbors,
// rounded up to a power of two (which also limits recompilations)
static int nbrs_buf_size(int max_n_nbrs)
{
    int buf_size = 64;
    while (buf_size < max_n_nbrs) buf_size *= 2;
    return buf_size;
}

static std::string build_options(int max_n_nbrs)
{
    std::ostringstream options;
    options << "-D MAX_NBRS=" << nbrs_buf_size(max_n_nbrs);
    return options.str();
}

// Work group size: each work item has its own array of drawn observations in private
// memory, so a work group of the usual size can need more of it than a device has
static size_t work_group_size(int max_n_nbrs)
{
    size_t local_item_size = 64;
    while (local_item_size > 1 &&
           local_item_size * sizeof(cl_int) * nbrs_buf_size(max_n_nbrs) > 65536) {
        local_item_size /= 2;
    }
    return local_item_size;
}

bool gpu_lisa(const char* cl_path, int rows, int permutations, unsigned long long last_seed_used, double* values, double* local_moran, GalElement* w, double* p)
{
    if (rows <= 0) return false;

    int max_n_nbrs = 0;
    int* num_nbrs = new int[rows];
    
    if (!prepare_num_nbrs(rows, w, true, num_nbrs, max_n_nbrs)) {
        delete[] num_nbrs;
        return false;
    }
    
    // Load the kernel source code into the array source_str
    std::ifstream t(cl_path);
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string src_code(buffer.str());

    char *source_str = strdup(src_code.c_str());
    size_t source_size = strlen(source_str);
    
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS) {
        delete[] num_nbrs;
        return false;
    }
    
    cl_uint maxDevices = 10;
    cl_device_id* devices = new cl_device_id[maxDevices];
    cl_uint nrDevices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, maxDevices, devices, &ret_num_devices);
    if (ret != CL_SUCCESS) {
        delete[] devices;
        delete[] num_nbrs;
        return false;
    }
	if (ret_num_devices==0) {
        delete[] devices;
        delete[] num_nbrs;
		return false;
	}
    cl_device_id device_id = devices[0];
    if (ret_num_devices==2) {
        device_id = devices[1];
    }
    //ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
    
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, ret_num_devices, devices, NULL, NULL, &ret);
    
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    
    // Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(double)*rows, NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(double)*rows, NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(int)*rows, NULL, &ret);
    cl_mem p_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(double)*rows, NULL, &ret);
    
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, sizeof(double)*rows,
                               values, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, sizeof(double)*rows,
                               local_moran, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, c_mem_obj, CL_TRUE, 0, sizeof(int)*rows,
                               num_nbrs, 0, NULL, NULL);
    // p is copied in: the kernel leaves the p-value of an isolate untouched
    ret = clEnqueueWriteBuffer(command_queue, p_mem_obj, CL_TRUE, 0, sizeof(double)*rows,
                               p, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        delete[] num_nbrs;
        
        ret = clReleaseMemObject(a_mem_obj);
        ret = clReleaseMemObject(b_mem_obj);
        ret = clReleaseMemObject(c_mem_obj);
        ret = clReleaseMemObject(p_mem_obj);
        
		return false;
	}
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, build_options(max_n_nbrs).c_str(), NULL, NULL);
    
	if (ret != CL_SUCCESS) {
        std::cout<<"Program Build failed\n";
        size_t length;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        std::cout<<"--- Build log ---\n "<<buffer<<endl;
        
        delete[] num_nbrs;
        
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(a_mem_obj);
        ret = clReleaseMemObject(b_mem_obj);
        ret = clReleaseMemObject(c_mem_obj);
        ret = clReleaseMemObject(p_mem_obj);
        
		return false;
	}

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "lisa", &ret);
    
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&rows);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&permutations);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_ulong), (void *)&last_seed_used);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&c_mem_obj);
    ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&p_mem_obj);
    
	if (ret != CL_SUCCESS) {
        delete[] num_nbrs;
        
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(a_mem_obj);
        ret = clReleaseMemObject(b_mem_obj);
        ret = clReleaseMemObject(c_mem_obj);
        ret = clReleaseMemObject(p_mem_obj);
        
		return false;
	}

    // Execute the OpenCL kernel on the list
    size_t local_item_size = work_group_size(max_n_nbrs);
    size_t global_item_size = local_item_size * ceil(rows/(double)local_item_size);
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                 &global_item_size, &local_item_size, 0, NULL, NULL);
    
    // Read the memory buffer C on the device to the local variable C
    if (ret == CL_SUCCESS) {
        ret = clEnqueueReadBuffer(command_queue, p_mem_obj, CL_TRUE, 0,
                                  sizeof(double) * rows, p, 0, NULL, NULL);
    }
    // a kernel that did not run must not be reported as a success: otherwise the
    // caller uses the p-values it passed in
    bool kernel_ok = ret == CL_SUCCESS;
    
    // Display the result to the screen
    //for(size_t i = 0; i < 20; i++)
        //printf("%f\n", p[i]);
    
    // Clean up
    //ret = clFlush(command_queue);
    //ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseMemObject(p_mem_obj);
    //ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

	if (ret != CL_SUCCESS) {
        delete[] num_nbrs;
		return false;
	}

    delete[] num_nbrs;
    delete[] devices;
	return kernel_ok;
}

bool gpu_localjoincount(const char* cl_path, int rows, int permutations, unsigned long long last_seed_used, int num_vars, int* zz, double* local_jc, GalElement* w, double* p)
{
    if (rows <= 0) return false;

    // num_vars is not passed to the kernel: the caller has combined the variables into zz
    int max_n_nbrs = 0;
    int* num_nbrs = new int[rows];
    
    // any observation can be drawn into a permutation, isolates included
    if (!prepare_num_nbrs(rows, w, false, num_nbrs, max_n_nbrs)) {
        delete[] num_nbrs;
        return false;
    }
    
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
    
    fp = fopen(cl_path, "r");
    if (!fp) {
        delete[] num_nbrs;
        fprintf(stderr, "Failed to load kernel.\n");
        return false;
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS) {
        delete[] num_nbrs;
        return false;
    }
    
    cl_uint maxDevices = 10;
    cl_device_id* devices = new cl_device_id[maxDevices];
    cl_uint nrDevices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, maxDevices, devices, &ret_num_devices);
    if (ret != CL_SUCCESS) {
        if (devices) delete[] devices;
        if(num_nbrs) delete[] num_nbrs;
        return false;
    }
    if (ret_num_devices==0) {
        if (devices) delete[] devices;
        if(num_nbrs) delete[] num_nbrs;
        return false;
    }
    cl_device_id device_id = devices[0];
    if (ret_num_devices==2) {
        device_id = devices[1];
    }
    //ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
    
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, ret_num_devices, devices, NULL, NULL, &ret);
    
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    
    
    // Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(int)*rows, NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(double)*rows, NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(int)*rows, NULL, &ret);
    cl_mem p_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(double)*rows, NULL, &ret);
    
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, sizeof(int)*rows, zz, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, sizeof(double)*rows, local_jc, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, c_mem_obj, CL_TRUE, 0, sizeof(int)*rows, num_nbrs, 0, NULL, NULL);
    // p is copied in: the kernel leaves the p-value of an isolate untouched
    ret = clEnqueueWriteBuffer(command_queue, p_mem_obj, CL_TRUE, 0, sizeof(double)*rows, p, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        if(num_nbrs) delete[] num_nbrs;
        return false;
    }
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char **)&source_str, (const size_t *)&source_size, &ret);
    
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, build_options(max_n_nbrs).c_str(), NULL, NULL);
    
    if (ret != CL_SUCCESS) {
        std::cout<<"Program Build failed\n";
        size_t length;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        std::cout<<"--- Build log ---\n "<<buffer<<endl;
        
        if(num_nbrs) delete[] num_nbrs;
        
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(a_mem_obj);
        ret = clReleaseMemObject(b_mem_obj);
        ret = clReleaseMemObject(c_mem_obj);
        ret = clReleaseMemObject(p_mem_obj);
        
        return false;
    }
    
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "localjc", &ret);
    
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&rows);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&permutations);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_ulong), (void *)&last_seed_used);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&c_mem_obj);
    ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&p_mem_obj);
    
    if (ret != CL_SUCCESS) {
        if(num_nbrs) delete[] num_nbrs;
        
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(a_mem_obj);
        ret = clReleaseMemObject(b_mem_obj);
        ret = clReleaseMemObject(c_mem_obj);
        ret = clReleaseMemObject(p_mem_obj);
        
        return false;
    }
    
    // Execute the OpenCL kernel on the list
    size_t local_item_size = work_group_size(max_n_nbrs);
    size_t global_item_size = local_item_size * ceil(rows/(double)local_item_size);
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    
    // Read the memory buffer C on the device to the local variable C
    if (ret == CL_SUCCESS) {
        ret = clEnqueueReadBuffer(command_queue, p_mem_obj, CL_TRUE, 0, sizeof(double) * rows, p, 0, NULL, NULL);
    }
    // a kernel that did not run must not be reported as a success: otherwise the
    // caller uses the p-values it passed in
    bool kernel_ok = ret == CL_SUCCESS;
    
    //Display the result to the screen
    //for(size_t i = 0; i < 20; i++) {
    //    printf("%f, %f, %f\n", values[i], values[1*rows + i], p[i]);
    //}

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseMemObject(p_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    
    if (ret != CL_SUCCESS) {
        if(num_nbrs) delete[] num_nbrs;
        return false;
    }
    
    if(num_nbrs) delete[] num_nbrs;
    
    return kernel_ok;
}
#endif
