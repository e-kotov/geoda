#ifndef __GEODA_CENTER_METAL_LISA_H___
#define __GEODA_CENTER_METAL_LISA_H___

#ifdef __APPLE__

#include <vector>

class GalElement;

// Apple Metal counterparts of gpu_lisa() and gpu_localjoincount() (OpenCL is
// deprecated on macOS and its fp64 kernels do not run on Apple Silicon GPUs).
// Conditional permutation follows AbstractCoordinator::CalcPseudoP_range(),
// with the random sequence of observation i starting at last_seed_used + i.
// Return false if Metal can't be used.
//
// Ties: Apple GPUs have no fp64, so the kernels add the drawn values in exact 128 bit
// fixed point where the CPU adds them in double.  Both count a permutation as larger
// when the permuted statistic is >= the observed one, and they agree exactly whenever
// the CPU's own arithmetic decides that comparison.  A permutation whose exact
// difference from the observed value is no larger than what the CPU's k additions and
// the few operations around them can lose -- (k + 4) * 2^-52 of the sum of the absolute
// values involved -- is always counted as larger here, while on the CPU the same
// permutation is decided by roundoff.  Only data with (near) ties is affected, and only
// by a few steps of 1/(permutations + 1).

bool is_metal_supported();

// One conditional permutation test of the Local Moran family (univariate, bivariate,
// median, undefined values, several time periods). The vectors are indexed by time
// period as in LisaCoordinator, and one draw serves all periods, as ComputeLarger() does.
struct GdaLisaPerm {
    int rows;
    int permutations;
    int num_time_vals;
    unsigned long long last_seed_used;
    bool using_median;                             // LisaCoordinator::using_median
    std::vector<double*> data1;                    // data1_vecs: multiplies the lag
    std::vector<double*> lagged;                   // data1_vecs, or data2_vecs when
                                                   // bivariate: lagged over the neighbors
    std::vector<double*> lags;                     // lags_vecs: the observed lag (mean or
                                                   // median) over the neighbors
    std::vector<GalElement*> w;                    // Gal_vecs[t]->gal
    const std::vector<std::vector<bool> >* undef;  // undef_tms, or NULL if there is none

    GdaLisaPerm() : rows(0), permutations(0), num_time_vals(0), last_seed_used(0),
                    using_median(false), undef(0) {}
};

// Fills p[t][i] of every observation that has neighbors; row-standardized weights only
bool metal_lisa(const char* metal_path, const GdaLisaPerm& perm, const std::vector<double*>& p);

// Univariate Local Moran of one time period without undefined values
bool metal_lisa(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                double* values, double* local_moran, GalElement* w, double* p);

// undef is JCCoordinator's undef_tms[t]: data undefs AND isolates, the observations its
// draw rejects and that get no pseudo p-value (MLJCCoordinator.cpp:252-263).  No default:
// a caller that passes nothing would get a draw that does not reject them.
bool metal_localjoincount(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                          int num_vars, int* zz, double* local_jc, GalElement* w, double* p,
                          const std::vector<bool>* undef);

#endif // __APPLE__

#endif // __GEODA_CENTER_METAL_LISA_H___
