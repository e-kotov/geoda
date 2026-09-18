#ifndef __GEODA_CENTER_METAL_LISA_H___
#define __GEODA_CENTER_METAL_LISA_H___

#ifdef __APPLE__

class GalElement;

// Apple Metal counterparts of gpu_lisa() and gpu_localjoincount() (OpenCL is
// deprecated on macOS and its fp64 kernels do not run on Apple Silicon GPUs).
// Conditional permutation follows AbstractCoordinator::CalcPseudoP_range(),
// with the random sequence of observation i starting at last_seed_used + i.
// Return false if Metal can't be used.

bool is_metal_supported();

bool metal_lisa(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                double* values, double* local_moran, GalElement* w, double* p);

bool metal_localjoincount(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                          int num_vars, int* zz, double* local_jc, GalElement* w, double* p);

#endif // __APPLE__

#endif // __GEODA_CENTER_METAL_LISA_H___
