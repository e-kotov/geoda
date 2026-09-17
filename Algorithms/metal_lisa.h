#ifndef __GEODA_CENTER_METAL_LISA_H___
#define __GEODA_CENTER_METAL_LISA_H___

#ifdef STANDALONE_TEST
#include <vector>
class GalElement {
    std::vector<long> nbr;
    std::vector<double> nbrWeight;
public:
    GalElement() {}
    void SetSizeNbrs(size_t sz) { nbr.resize(sz); nbrWeight.resize(sz, 1.0); }
    void SetNbr(size_t pos, long n) { if (pos < nbr.size()) nbr[pos] = n; }
    long Size() const { return (long)nbr.size(); }
    long operator[](size_t n) const { return nbr[n]; }
};
#else
#include "../ShapeOperations/GalWeight.h"
#endif

#ifdef __APPLE__

bool is_metal_supported();

bool metal_lisa(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                double* values, double* local_moran, GalElement* w, double* p);

bool metal_localjoincount(const char* metal_path, int rows, int permutations, unsigned long long last_seed_used,
                          int num_vars, int* zz, double* local_jc, GalElement* w, double* p);

#endif // __APPLE__

#endif // __GEODA_CENTER_METAL_LISA_H___
