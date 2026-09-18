// Standalone test of the Apple Metal permutation kernels against the CPU algorithm:
//
//   clang++ -std=c++14 -O2 -fobjc-arc Algorithms/test_metal_lisa.mm \
//       -framework Metal -framework Foundation -o test_metal_lisa
//   ./test_metal_lisa Algorithms/lisa_kernel.metal Algorithms/localjc_kernel.metal
//
// The reference below is the conditional permutation of
// AbstractCoordinator::CalcPseudoP_range() / JCCoordinator::CalcPseudoP_range().
// 1. With the random sequence of observation i starting at seed + i (what the GPU
//    does), GPU and CPU draw the same permutations, so pseudo p-values must be equal --
//    except for permutations so close to the observed value that the CPU's own double
//    arithmetic decides its >= by roundoff.  Those the GPU always counts as larger, so
//    its count must lie between the CPU's count with none and with all of them counted
//    ("the BOUNDS rule", see the derivation further down).  Local Join Count sums small
//    integers and is therefore bit-identical.
// 2. With one sequential random sequence (what single threaded GeoDa does), pseudo
//    p-values must agree within Monte Carlo error.
// 3. No pseudo p-value may be written where the CPU runs no permutation test.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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
#include "test_data_guerry.h"
#include "test_data_natregimes.h"

// Gda::ThomasWangHashDouble()
static double ThomasWangHashDouble(uint64_t key)
{
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return 5.42101086242752217E-20 * key;
}

// Knuth's TwoSum: a + b == s + err exactly.  The tie test classifies by the EXACT
// difference of the permuted and the observed value: the CPU's rounding is what the
// tolerance stands for, so it must not decide which side of the tolerance a permutation
// falls on (it does decide the >= that the tolerance then overrides).
static double two_sum_err(double a, double b, double s)
{
    double t = s - a;
    return (a - (s - t)) + (b - t);
}

// Neumaier's compensated summation: sum + comp is the exact sum up to 2^-105 of it
struct ExactSum {
    double sum, comp;
    ExactSum() : sum(0), comp(0) {}
    void add(double v) {
        double s = sum + v;
        comp += fabs(sum) >= fabs(v) ? (sum - s) + v : (v - s) + sum;
        sum = s;
    }
    double minus(double hi, double lo) const { return (sum - hi) + (comp - lo); } // from hi + lo
};

// the two values GenUtils::Median() averages (the same one twice for an odd count)
static void median_pair(std::vector<double> data, double& lo, double& hi)
{
    lo = hi = 0;
    if (data.empty()) return;
    std::sort(data.begin(), data.end());
    lo = data[(data.size() - 1) / 2];
    hi = data[data.size() / 2];
}

// ---------------------------------------------------------------------------
// How the GPU may differ from the CPU, and how that is tested (the BOUNDS rule)
//
// Both count a permutation as larger when the permuted statistic is >= the observed
// one.  The kernel adds the drawn values in exact 128 bit fixed point, the CPU adds
// them in double, so the two can only disagree about permutations whose EXACT
// difference from the observed value is small enough for the CPU's own rounding to
// decide its >=.  Every permutation is classified with
//   diff  the exact difference (compensated summation and TwoSum: nothing is rounded), and
//   E     an upper bound on the error of the CPU's double evaluation, derived below from
//         the CPU's operations alone and NOT from the kernel's tolerance.
// The kernel counts every permutation inside the band it documents as larger, so the
// permutations whose outcome is not pinned down are those with |diff| <= max(E, band).
// The GPU's count must lie between the count with none of them counted and the count
// with all of them counted (the BOUNDS rule); outside that set it must reproduce the
// CPU exactly.  A separate assertion checks band >= E for every permutation of every
// fixture, which is what makes the width of the documented band a tested property and
// not an assumption: were the band ever narrower than the CPU's own uncertainty, the
// GPU could contradict a decision the CPU is entitled to make.  (The band is about 2x
// E, and E itself is conservative: measured over 70 million permutations of these
// fixtures, the CPU's double evaluation never disagreed with the exact comparison at
// |diff| above 0.024 E.  What the GPU does strictly inside the band is constrained by
// the exact_ties() fixture instead, where the CPU's own arithmetic has no error at all.)
//
// Derivation of E for the mean (u = 2^-53; every rounding is fl(z) = z(1 + d), |d| <= u).
// The CPU computes  permuted = fl(fl(fl_sum(v_1..v_k) / k) * x_i)  and compares it with
// observed = fl(x_i * lag_i)  (LisaCoordinator.cpp:673-676 and :534).  With
//   S = v_1 + ... + v_k exactly and |fl_sum - S| <= 1.05 (k-1) u sumAbs  (the textbook
//     (k-1)u/(1-(k-1)u) bound; the kernel refuses k >= 2^16, so the factor is < 1.00001),
//   D = S - k lag_i, the exact difference the kernel works with,
// dividing the comparison by x_i (which only reverses it when x_i < 0) gives
//   permuted >= observed  <=>  D + e_sum + k lag_i (d_div + d_mul - d_obs)
//                                + D (d_div + d_mul) + O(u^2) >= 0,
// so the sign of D decides the comparison unless
//   |D| <= 1.05 (k-1) u sumAbs + 3u |k lag_i| + 2u|D| + O(u^2).
// In that case |D| is itself of order u, so 2u|D| and the second order terms are many
// orders below the 3.15 u sumAbs that widening (k-1) to (k+2) adds.  The legacy entry
// point metal_lisa(values, local_moran, w) recovers lag_i as local_moran[i]/values[i],
// one more rounding of the observed side: obs_roundings is 4 there, 3 on the
// coordinator's path, where lags_vecs[t][i] is handed to the GPU as it is.
static const double kUnit = ldexp(1.0, -53);

static double cpu_error_mean(int k, double sum_abs, double obs_sum, int obs_roundings)
{
    return kUnit * (1.05 * (k + 2) * sum_abs + obs_roundings * fabs(obs_sum));
}

// The median compares fl(fl((v_lo+v_hi)/2) * x_i) with fl(x_i * fl((o_lo+o_hi)/2)):
// halving is exact, so there are two roundings on each side of the exact difference
// D = (v_lo + v_hi) - (o_lo + o_hi), i.e. E = 2u(|v_lo| + |v_hi| + |o_lo+o_hi|) + O(u^2).
static double cpu_error_median(double sum_abs, double obs_sum)
{
    return 2.1 * kUnit * (sum_abs + fabs(obs_sum));
}

// The band lisa_kernel.metal documents and always counts as ">=" (tol128(), and
// tol128(.., 2) for the median).  Asserted to be at least cpu_error_*() everywhere.
static double kernel_band(int k, double sum_abs, double obs_sum)
{
    return ldexp(1.0, -52) * (k + 4) * (sum_abs + fabs(obs_sum));
}

// AbstractCoordinator.cpp:568 folds a count into the smaller tail.  The fold is unimodal
// in the count, so the counts of an interval fold onto a contiguous range of p-values.
static int fold_count(int c, int permutations)
{
    return (permutations - c <= c) ? permutations - c : c;
}

// What the GPU is allowed to return, per time period and observation, plus the
// bookkeeping of the rule.  p_lo < 0 marks an observation without a permutation test.
struct TieBounds {
    std::vector<std::vector<double> > p_lo, p_hi;
    std::vector<std::vector<int> > near;   // permutations the CPU's doubles cannot decide
    int n_exact;                           // permutations whose exact difference is 0
    int band_too_narrow;                   // permutations where the documented band < E
    TieBounds() : n_exact(0), band_too_narrow(0) {}
    void init(int T, int n) {
        p_lo.assign(T, std::vector<double>(n, -1.0));
        p_hi = p_lo;
        near.assign(T, std::vector<int>(n, 0));
    }
    void set(int t, int i, int count_lo, int count_hi, int permutations) {
        int mid = permutations / 2;                       // where the fold is largest
        if (mid < count_lo) mid = count_lo;
        if (mid > count_hi) mid = count_hi;
        int fmin = std::min(fold_count(count_lo, permutations),
                            fold_count(count_hi, permutations));
        p_lo[t][i] = (fmin + 1.0) / (permutations + 1);
        p_hi[t][i] = (fold_count(mid, permutations) + 1.0) / (permutations + 1);
    }
};

// CPU reference for both Local Moran (is_jc = false, row-standardized weights) and
// Local Join Count (is_jc = true)
static void cpu_reference(bool is_jc, bool per_obs_seed, int n, int permutations, uint64_t seed,
                          const std::vector<double>& x, const std::vector<double>& local_sa,
                          const std::vector<GalElement>& w, std::vector<double>& p,
                          TieBounds* bounds = 0, const std::vector<bool>* undef = 0)
{
    int max_rand = n - 1;
    uint64_t seed_start = seed;
    if (bounds) bounds->init(1, n);
    std::vector<bool> drawn(n, false);
    std::vector<int> perm_nbrs;

    for (int i = 0; i < n; i++) {
        if (per_obs_seed) seed_start = seed + i;
        if (undef && (*undef)[i]) continue;          // JCCoordinator::CalcPseudoP_range():623
        if (is_jc && local_sa[i] == 0) { p[i] = 0; continue; }
        int num_nbrs = (int)w[i].Size();
        if (w[i].Check(i)) num_nbrs -= 1;
        if (num_nbrs == 0) continue;                 // no permutation test, no p-value

        int count_larger = 0, count_lo = 0, count_hi = 0;
        for (int perm = 0; perm < permutations; perm++) {
            perm_nbrs.clear();
            while ((int)perm_nbrs.size() < num_nbrs) {
                double rng_val = ThomasWangHashDouble(seed_start++) * max_rand;
                int r = (int)(rng_val < 0.0 ? ceil(rng_val - 0.5) : floor(rng_val + 0.5));
                // Join Count rejects undefined observations, Local Moran neighborless ones
                if (r != i && !drawn[r] &&
                    (is_jc ? (!undef || !(*undef)[r]) : w[r].Size() > 0)) {
                    drawn[r] = true;
                    perm_nbrs.push_back(r);
                }
            }
            double lag = 0, lag_abs = 0;
            ExactSum exact;
            for (int j = num_nbrs - 1; j >= 0; j--) { // GeoDaSet::Pop() order
                lag += x[perm_nbrs[j]];
                exact.add(x[perm_nbrs[j]]);
                lag_abs += fabs(x[perm_nbrs[j]]);
                drawn[perm_nbrs[j]] = false;
            }
            if (is_jc) {
                if (lag >= local_sa[i]) count_larger++;   // small integers: exact
                continue;
            }
            double permuted = lag / num_nbrs * x[i];
            bool counted = permuted >= local_sa[i];      // LisaCoordinator::ComputeLarger()
            if (counted) count_larger++;
            // the exact difference between the drawn sum and the sum the observed lag
            // stands for, the observed side as metal_lisa()'s legacy entry point derives it
            double obs_lag = x[i] == 0 ? 0 : local_sa[i] / x[i];
            double obs_sum = obs_lag * num_nbrs, obs_err = fma(obs_lag, (double)num_nbrs, -obs_sum);
            double diff = exact.minus(obs_sum, obs_err);
            double e_cpu = cpu_error_mean(num_nbrs, lag_abs, obs_sum, 4);
            double band = kernel_band(num_nbrs, lag_abs, obs_sum);
            if (bounds) {
                if (e_cpu > band) bounds->band_too_narrow++;
                if (diff == 0) bounds->n_exact++;
            }
            // x[i] == 0 makes both compared values exactly 0: both count, nothing is undecided
            bool near = x[i] != 0 && fabs(diff) <= std::max(e_cpu, band);
            if (near && bounds) bounds->near[0][i]++;
            count_lo += (!near && counted);
            count_hi += (near || counted);
        }
        if (bounds && !is_jc) bounds->set(0, i, count_lo, count_hi, permutations);
        if (is_jc ? (permutations - count_larger < count_larger) : (permutations - count_larger <= count_larger)) {
            count_larger = permutations - count_larger;
        }
        p[i] = (count_larger + 1.0) / (permutations + 1);
    }
}

struct TestCase {
    const char* name;
    std::vector<double> x;   // standardized variable
    std::vector<int> zz;     // binary variable
    std::vector<GalElement> w;
    bool exact_arith;        // the CPU's own double arithmetic is exact (see exact_ties())
    TestCase() : name(""), exact_arith(false) {}
};

static void standardize(std::vector<double>& x)
{
    double mean = 0, ssd = 0;
    for (double v : x) mean += v;
    mean /= x.size();
    for (double v : x) ssd += (v - mean) * (v - mean);
    double sd = sqrt(ssd / (x.size() - 1));
    if (!(sd > 0)) { printf("[FAIL] constant or invalid test data\n"); exit(1); }
    for (double& v : x) v = (v - mean) / sd;
}

// GeoDa sample data, see make_test_data.R
static TestCase sample_data(const char* name, int n, const double* x, const int* nbr_offset, const int* nbrs)
{
    TestCase tc;
    tc.name = name;
    tc.x.assign(x, x + n);
    standardize(tc.x);
    tc.w.resize(n);
    for (int i = 0; i < n; i++) {
        int k = nbr_offset[i + 1] - nbr_offset[i];
        tc.w[i].SetSizeNbrs(k);
        for (int j = 0; j < k; j++) tc.w[i].SetNbr(j, nbrs[nbr_offset[i] + j]);
    }
    for (double v : tc.x) tc.zz.push_back(v > 0);
    return tc;
}

// rook lattice with tied integer values, isolates and a self-neighbor
static TestCase lattice(int side)
{
    TestCase tc;
    tc.name = "50x50 lattice, rook, isolates + self-neighbor";
    int n = side * side;
    tc.w.resize(n);
    for (int i = 0; i < n; i++) {
        tc.x.push_back((double)(ThomasWangHashDouble(i) < 0.3 ? 5 : (i / side + i % side) % 7));
        int r = i / side, c = i % side;
        std::vector<long> nb;
        if (r > 0) nb.push_back(i - side);
        if (r < side - 1) nb.push_back(i + side);
        if (c > 0) nb.push_back(i - 1);
        if (c < side - 1) nb.push_back(i + 1);
        if (i == 7) nb.push_back(i);              // self-neighbor
        if (i % 97 == 0) nb.clear();              // isolate (asymmetric weights are fine here)
        tc.w[i].SetSizeNbrs(nb.size());
        for (size_t j = 0; j < nb.size(); j++) tc.w[i].SetNbr(j, nb[j]);
    }
    for (double v : tc.x) tc.zz.push_back(v == 5);
    standardize(tc.x);
    return tc;
}

// more than 64 neighbors (kernel work array is resized), unstandardized values of large
// magnitude with exact zeros and ties, one observation linked to all others
static TestCase dense(int n, int k)
{
    TestCase tc;
    tc.name = "dense weights, 100 neighbors, unstandardized";
    tc.w.resize(n);
    for (int i = 0; i < n; i++) {
        double v = floor(ThomasWangHashDouble(i + 1000) * 20) - 5;
        tc.x.push_back(v * 1e6);
        tc.zz.push_back(v > 8);
        int ki = (i == 0) ? n - 1 : k;
        tc.w[i].SetSizeNbrs(ki);
        for (int j = 0; j < ki; j++) tc.w[i].SetNbr(j, (i + 1 + j) % n);
    }
    return tc;
}

// few distinct values that are not representable once standardized, many neighbors:
// almost every permutation of some observations ties with the observed value
static TestCase tied_dense(int n, int k)
{
    TestCase tc;
    tc.name = "tied values, 150 neighbors, standardized";
    tc.w.resize(n);
    for (int i = 0; i < n; i++) {
        double v = floor(ThomasWangHashDouble(i + 5000) * 3);
        tc.x.push_back(v == 2 ? 0.1 : v);
        tc.zz.push_back(v == 1);
        tc.w[i].SetSizeNbrs(k);
        for (int j = 0; j < k; j++) tc.w[i].SetNbr(j, (i + 1 + 7 * j) % n);
    }
    standardize(tc.x);
    return tc;
}

// Small integers, four neighbors and many exact zeros.  Sums of at most 4 integers below
// 2^15, their quotient by the power of two 4 and the products with x[i] are all exactly
// representable, so the CPU's own arithmetic has NO rounding error here: ties are decided
// by exact arithmetic and the GPU must reproduce the CPU bit for bit, including the
// permutations that tie exactly (which is what ">=" means) and those where the kernel's
// tolerance itself is zero (all drawn values and the observed lag are 0).  This is the
// fixture that constrains what the GPU does INSIDE the tolerance band; the BOUNDS rule
// deliberately does not.
static TestCase exact_ties(int n)
{
    TestCase tc;
    tc.name = "exact integer arithmetic, 4 neighbors, many zeros";
    tc.exact_arith = true;
    tc.w.resize(n);
    for (int i = 0; i < n; i++) {
        double h = ThomasWangHashDouble(i + 90000) * 7;
        double v = h < 4 ? 0 : floor(h) - 5;      // 0 with probability 4/7, else -2..1
        tc.x.push_back(v);
        tc.zz.push_back(v > 0);
        tc.w[i].SetSizeNbrs(4);                   // a power of two: lag/4 is exact
        for (int j = 0; j < 4; j++) tc.w[i].SetNbr(j, (i + 1 + 3 * j) % n);
    }
    return tc;
}

static int failures = 0;

static bool check(bool ok, const char* what)
{
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) failures++;
    return ok;
}

static bool check(bool ok, const std::string& what) { return check(ok, what.c_str()); }

// The tie assertions of one GPU result (see the BOUNDS rule above)
static void check_tie_bounds(const std::string& name, const TieBounds& b,
                             const std::vector<std::vector<double> >& p_cpu,
                             const std::vector<std::vector<double> >& p_gpu, bool exact_arith)
{
    int tested = 0, undecided = 0, n_diff = 0, out_of_bounds = 0, n_diff_all = 0;
    for (size_t t = 0; t < b.p_lo.size(); t++) {
        for (size_t i = 0; i < b.p_lo[t].size(); i++) {
            if (b.p_lo[t][i] < 0) continue;              // no permutation test
            tested++;
            n_diff_all += p_gpu[t][i] != p_cpu[t][i];
            if (b.near[t][i] > 0) undecided++;
            else n_diff += p_gpu[t][i] != p_cpu[t][i];
            out_of_bounds += p_gpu[t][i] < b.p_lo[t][i] || p_gpu[t][i] > b.p_hi[t][i];
        }
    }
    printf("  %d of %d p-values have a permutation the CPU's doubles cannot decide\n",
           undecided, tested);
    check(n_diff == 0, name + ": identical to the CPU where every permutation is decidable");
    check(out_of_bounds == 0, name + ": within the bounds the CPU's own rounding allows");
    check(b.band_too_narrow == 0, name + ": the documented band covers the CPU's error bound");
    if (exact_arith) {
        check(n_diff_all == 0, name + ": exact arithmetic, so identical to the CPU everywhere");
        check(b.n_exact > 0, name + ": the fixture does produce exactly tied permutations");
    }
}

static std::vector<std::vector<double> > one_period(const std::vector<double>& v)
{
    return std::vector<std::vector<double> >(1, v);
}

static void run(TestCase& tc, const char* lisa_path, const char* jc_path, int permutations)
{
    const uint64_t seed = 123456789; // GeoDa's default seed
    const double untouched = -1;
    int n = (int)tc.x.size();
    printf("%s (n=%d, permutations=%d)\n", tc.name, n, permutations);

    // JCCoordinator marks isolates undefined (MLJCCoordinator.cpp:252-263): its draw
    // rejects them and they get no pseudo p-value
    std::vector<bool> isolate(n, false);
    for (int i = 0; i < n; i++) isolate[i] = tc.w[i].Size() == 0;

    for (int is_jc = 0; is_jc <= 1; is_jc++) {
        // observed statistics
        std::vector<double> x = tc.x, local_sa(n, 0);
        if (is_jc) x.assign(tc.zz.begin(), tc.zz.end());
        for (int i = 0; i < n; i++) {
            double lag = 0;
            int num_nbrs = 0;
            for (long j = 0; j < tc.w[i].Size(); j++) {
                if (tc.w[i][j] != i) { lag += x[tc.w[i][j]]; num_nbrs++; }
            }
            if (num_nbrs > 0) local_sa[i] = is_jc ? x[i] * lag : x[i] * lag / num_nbrs;
        }

        std::vector<double> p_gpu(n, untouched), p_cpu(n, untouched), p_seq(n, untouched);
        TieBounds bounds;
        const std::vector<bool>* jc_undef = is_jc ? &isolate : 0;
        bool ok = is_jc ? metal_localjoincount(jc_path, n, permutations, seed, 1, tc.zz.data(), local_sa.data(), tc.w.data(), p_gpu.data(), jc_undef)
                        : metal_lisa(lisa_path, n, permutations, seed, x.data(), local_sa.data(), tc.w.data(), p_gpu.data());
        check(ok, is_jc ? "metal_localjoincount() runs" : "metal_lisa() runs");
        if (!ok) continue;
        cpu_reference(is_jc, true, n, permutations, seed, x, local_sa, tc.w, p_cpu, &bounds, jc_undef);
        cpu_reference(is_jc, false, n, permutations, seed, x, local_sa, tc.w, p_seq, 0, jc_undef);

        // a pseudo p-value exactly where the CPU computes one, and nowhere else
        int wrong_slot = 0;
        for (int i = 0; i < n; i++) {
            int num_nbrs = (int)tc.w[i].Size();
            if (tc.w[i].Check(i)) num_nbrs -= 1;
            if (is_jc ? isolate[i] : num_nbrs == 0) wrong_slot += p_gpu[i] != untouched;
            else if (is_jc && local_sa[i] == 0) wrong_slot += p_gpu[i] != 0.0;   // :625
            else wrong_slot += !(p_gpu[i] > 0 && p_gpu[i] <= 1);
        }
        check(wrong_slot == 0, "a pseudo p-value exactly where the CPU computes one");

        int n_diff = 0;
        double max_diff_seq = 0;
        for (int i = 0; i < n; i++) {
            if (p_gpu[i] == untouched) continue;
            n_diff += p_gpu[i] != p_cpu[i];
            max_diff_seq = std::max(max_diff_seq, fabs(p_gpu[i] - p_seq[i]));
        }
        if (is_jc) {
            check(n_diff == 0, "identical to CPU");
        } else {
            check_tie_bounds(tc.name, bounds, one_period(p_cpu), one_period(p_gpu), tc.exact_arith);
        }
        // difference of two independent Monte Carlo estimates, 6 standard errors
        printf("  sequential seed: max p-value diff %g\n", max_diff_seq);
        check(max_diff_seq <= 6 * sqrt(0.5 / permutations), "agrees with sequentially seeded CPU within Monte Carlo error");
    }
}

// Local Join Count with undefined values.  JCCoordinator keeps TWO masks and they are
// not the same: the data undefs zero zz[] (MLJCCoordinator.cpp:375-381) and prune the
// weights (GalWeight::Update(), :333-341), while the draw rejection and the "no p-value"
// test use the data undefs UNION the isolates of the original weights (:252-263, read at
// :623 and :649).
static void run_jc_undef(const TestCase& tc, const char* jc_path, int permutations)
{
    const uint64_t seed = 123456789;
    const double untouched = -1;
    const int n = (int)tc.x.size();
    printf("Local Join Count, undefined values (n=%d, permutations=%d)\n", n, permutations);

    std::vector<bool> data_undef(n, false), draw_undef(n, false);
    for (int i = 0; i < n; i++) {
        data_undef[i] = (i % 11 == 0);
        draw_undef[i] = data_undef[i] || tc.w[i].Size() == 0;
    }
    std::vector<int> zz(tc.zz);
    std::vector<double> zz_d(n), local_jc(n, 0), p_gpu(n, untouched), p_cpu(n, untouched);
    std::vector<GalElement> w(tc.w);
    for (int i = 0; i < n; i++) {
        if (data_undef[i]) zz[i] = 0;                // MLJCCoordinator.cpp:379-381
        zz_d[i] = zz[i];
    }
    for (int i = 0; i < n; i++) {                    // GalWeight::Update(), data undefs only
        std::vector<long> keep;
        for (long j = 0; j < tc.w[i].Size(); j++)
            if (!data_undef[tc.w[i][j]]) keep.push_back(tc.w[i][j]);
        w[i].SetSizeNbrs(keep.size());
        for (size_t j = 0; j < keep.size(); j++) w[i].SetNbr(j, keep[j]);
    }
    for (int i = 0; i < n; i++) {                    // :392-403
        if (zz[i] == 0) continue;
        for (long j = 0; j < w[i].Size(); j++)
            if (w[i][j] != i) local_jc[i] += zz[w[i][j]];
    }

    bool ok = metal_localjoincount(jc_path, n, permutations, seed, 1, zz.data(),
                                   local_jc.data(), w.data(), p_gpu.data(), &draw_undef);
    if (!check(ok, "metal_localjoincount() runs")) return;
    cpu_reference(true, true, n, permutations, seed, zz_d, local_jc, w, p_cpu, 0, &draw_undef);
    int n_diff = 0, wrong_slot = 0;
    for (int i = 0; i < n; i++) {
        if (draw_undef[i]) wrong_slot += p_gpu[i] != untouched;
        else if (local_jc[i] == 0) wrong_slot += p_gpu[i] != 0.0;
        else wrong_slot += !(p_gpu[i] > 0 && p_gpu[i] <= 1);
        if (p_gpu[i] != untouched) n_diff += p_gpu[i] != p_cpu[i];
    }
    check(wrong_slot == 0, "a pseudo p-value exactly where the CPU computes one");
    check(n_diff == 0, "identical to CPU");
}

// ---------------------------------------------------------------------------
// The other Local Moran variants: one draw serves every time period, the lag can come
// from a second variable (bivariate) or be a median, and undefined neighbors are left
// out of it, which changes the number of neighbors the lag is divided by
// ---------------------------------------------------------------------------
struct VariantCase {
    std::string name;
    int n, permutations, num_time_vals;
    uint64_t seed;
    bool using_median;
    std::vector<std::vector<double> > data1;   // multiplies the lag, LisaCoordinator::data1_vecs
    std::vector<std::vector<double> > lagged;  // lagged over the drawn neighbors
    std::vector<std::vector<double> > lags;    // observed lag, LisaCoordinator::lags_vecs
    std::vector<std::vector<bool> > undef;     // undef_tms
    std::vector<std::vector<GalElement> > w;   // Gal_vecs[t]: no links to undefined values
};

// AbstractCoordinator::CalcPseudoP_range() with LisaCoordinator::ComputeLarger() for all
// of them, plus the bounds the GPU's counts must respect (see the BOUNDS rule above).
static void cpu_variant_reference(const VariantCase& vc, std::vector<std::vector<double> >& p,
                                  TieBounds& bounds)
{
    const int n = vc.n, T = vc.num_time_vals, permutations = vc.permutations;
    p.assign(T, std::vector<double>(n, -1.0));   // sentinel: no permutation test
    bounds.init(T, n);
    std::vector<bool> drawn(n, false);
    std::vector<int> perm_nbrs;

    for (int i = 0; i < n; i++) {
        uint64_t seed_start = vc.seed + i;
        int num_nbrs = 0;
        for (int t = 0; t < T; t++) { // the largest count over the periods, upstream quirks included
            if (vc.w[t][i].Size() > num_nbrs) {
                num_nbrs = (int)vc.w[t][i].Size();
                if (vc.w[t][i].Check(i)) num_nbrs -= 1;
            }
        }
        if (num_nbrs == 0) continue;

        std::vector<int> count(T, 0), count_lo(T, 0), count_hi(T, 0);
        for (int perm = 0; perm < permutations; perm++) {
            perm_nbrs.clear();
            while ((int)perm_nbrs.size() < num_nbrs) {
                double rng_val = ThomasWangHashDouble(seed_start++) * (n - 1);
                int r = (int)(rng_val < 0.0 ? ceil(rng_val - 0.5) : floor(rng_val + 0.5));
                // the rejection reads the LAST period's weights, as the CPU's loop leaves them
                if (r != i && !drawn[r] && vc.w[T - 1][r].Size() > 0) {
                    drawn[r] = true;
                    perm_nbrs.push_back(r);
                }
            }
            for (int t = 0; t < T; t++) {
                const std::vector<double>& x = vc.data1[t];
                const std::vector<double>& a = vc.lagged[t];
                double lag = 0, lag_abs = 0, permuted = 0, diff = 0;
                double sum_abs = 0, obs_sum = 0, e_cpu = 0;
                int valid = 0;
                ExactSum exact;
                std::vector<double> nbr_data;
                for (int j = num_nbrs - 1; j >= 0; j--) { // GeoDaSet::Pop() order
                    int nb = perm_nbrs[j];
                    if (vc.undef[t][nb]) continue;
                    lag += a[nb];
                    exact.add(a[nb]);
                    lag_abs += fabs(a[nb]);
                    nbr_data.push_back(a[nb]);
                    valid++;
                }
                if (vc.using_median) {
                    double v_lo = 0, v_hi = 0, o_lo = 0, o_hi = 0;
                    median_pair(nbr_data, v_lo, v_hi);
                    if (!vc.undef[t][i]) {   // Calc() leaves the lag of an undefined value at 0
                        std::vector<double> obs_data;
                        for (long j = 0; j < vc.w[t][i].Size(); j++)
                            if (vc.w[t][i][j] != i) obs_data.push_back(a[vc.w[t][i][j]]);
                        median_pair(obs_data, o_lo, o_hi);
                    }
                    permuted = 0.5 * (v_lo + v_hi) * x[i];
                    double perm2 = v_lo + v_hi, obs2 = o_lo + o_hi;
                    sum_abs = fabs(v_lo) + fabs(v_hi);
                    obs_sum = obs2;
                    diff = (perm2 - obs2) + (two_sum_err(v_lo, v_hi, perm2) - two_sum_err(o_lo, o_hi, obs2));
                    e_cpu = cpu_error_median(sum_abs, obs2);
                } else {
                    permuted = (valid > 0 ? lag / valid : 0) * x[i];
                    sum_abs = lag_abs;
                    obs_sum = vc.lags[t][i] * valid;
                    diff = exact.minus(obs_sum, fma(vc.lags[t][i], (double)valid, -obs_sum));
                    // the coordinator hands lags_vecs[t][i] to the GPU as it is: 3 roundings
                    e_cpu = cpu_error_mean(valid, lag_abs, obs_sum, 3);
                }
                const double observed = x[i] * vc.lags[t][i];       // local_moran_vecs[t][i]
                const bool counted = permuted >= observed;
                if (counted) count[t]++;
                // x[i] == 0, and an all-undefined draw (the CPU does not divide and
                // compares 0 with the observed value), are exact: nothing is undecided
                bool near = false;
                if (x[i] != 0 && valid > 0) {
                    double band = kernel_band(vc.using_median ? 2 : valid, sum_abs, obs_sum);
                    if (e_cpu > band) bounds.band_too_narrow++;
                    if (diff == 0) bounds.n_exact++;
                    near = fabs(diff) <= std::max(e_cpu, band);
                }
                if (near) bounds.near[t][i]++;
                count_lo[t] += (!near && counted);
                count_hi[t] += (near || counted);
            }
            for (int j = 0; j < num_nbrs; j++) drawn[perm_nbrs[j]] = false;
        }
        for (int t = 0; t < T; t++) {
            int c = count[t];
            if (permutations - c <= c) c = permutations - c;
            p[t][i] = (c + 1.0) / (permutations + 1);
            bounds.set(t, i, count_lo[t], count_hi[t], permutations);
        }
    }
}

// LisaCoordinator::StandardizeData() and Calc() for one variant of a test case: undefined
// values every undef_every observations (isolates are undefined too), a second variable
// and time periods that permute the data
static VariantCase build_variant(const TestCase& tc, const std::string& name, int num_time_vals,
                                 bool bivariate, bool median, int undef_every,
                                 int permutations, uint64_t seed)
{
    const int n = (int)tc.x.size(), T = num_time_vals;
    VariantCase vc;
    vc.name = name;
    vc.n = n;
    vc.permutations = permutations;
    vc.num_time_vals = T;
    vc.seed = seed;
    vc.using_median = median;
    vc.undef.assign(T, std::vector<bool>(n, false));
    vc.w.assign(T, tc.w);
    vc.lags.assign(T, std::vector<double>(n, 0.0));

    for (int t = 0; t < T; t++) {
        std::vector<double> d1(n), d2(n);
        for (int i = 0; i < n; i++) {
            d1[i] = tc.x[(i + 3 * t) % n] + 0.25 * t * tc.x[i];
            d2[i] = 0.5 * tc.x[(i + n / 3) % n] - tc.x[i] + 1.5;     // a different shape
            vc.undef[t][i] = (undef_every && (i + t) % undef_every == 0) || tc.w[i].Size() == 0;
        }
        standardize(d1);
        standardize(d2);
        vc.data1.push_back(d1);
        vc.lagged.push_back(bivariate ? d2 : d1);
        // GalWeight::Update(): drop the links to undefined observations
        for (int i = 0; i < n; i++) {
            std::vector<long> keep;
            for (long j = 0; j < tc.w[i].Size(); j++)
                if (!vc.undef[t][tc.w[i][j]]) keep.push_back(tc.w[i][j]);
            vc.w[t][i].SetSizeNbrs(keep.size());
            for (size_t j = 0; j < keep.size(); j++) vc.w[t][i].SetNbr(j, keep[j]);
        }
        // observed lag over the non-self neighbors, mean or median
        const std::vector<double>& a = vc.lagged[t];
        for (int i = 0; i < n; i++) {
            if (vc.undef[t][i]) continue;
            std::vector<double> nbr_data;
            for (long j = 0; j < vc.w[t][i].Size(); j++)
                if (vc.w[t][i][j] != i) nbr_data.push_back(a[vc.w[t][i][j]]);
            if (nbr_data.empty()) continue;
            if (median) {
                double lo = 0, hi = 0;
                median_pair(nbr_data, lo, hi);
                vc.lags[t][i] = 0.5 * (lo + hi);
            } else {
                double lag = 0;
                for (size_t j = 0; j < nbr_data.size(); j++) lag += nbr_data[j];
                vc.lags[t][i] = lag / nbr_data.size();
            }
        }
    }
    return vc;
}

static void run_variant(VariantCase& vc, const char* lisa_path)
{
    const int n = vc.n, T = vc.num_time_vals;
    printf("%s (n=%d, periods=%d, permutations=%d)\n", vc.name.c_str(), n, T, vc.permutations);

    const double untouched = -1;
    std::vector<std::vector<double> > p_cpu, p_gpu(T, std::vector<double>(n, untouched));
    TieBounds bounds;
    cpu_variant_reference(vc, p_cpu, bounds);

    GdaLisaPerm perm;
    perm.rows = n;
    perm.permutations = vc.permutations;
    perm.num_time_vals = T;
    perm.last_seed_used = vc.seed;
    perm.using_median = vc.using_median;
    perm.undef = &vc.undef;
    std::vector<double*> p(T);
    for (int t = 0; t < T; t++) {
        perm.data1.push_back(vc.data1[t].data());
        perm.lagged.push_back(vc.lagged[t].data());
        perm.lags.push_back(vc.lags[t].data());
        perm.w.push_back(vc.w[t].data());
        p[t] = p_gpu[t].data();
    }
    if (!check(metal_lisa(lisa_path, perm, p), "metal_lisa() runs")) return;

    // a pseudo p-value exactly where the CPU computes one, and nowhere else
    int wrong_slot = 0;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++)
            wrong_slot += (p_cpu[t][i] < 0) ? (p_gpu[t][i] != untouched)
                                            : !(p_gpu[t][i] > 0 && p_gpu[t][i] <= 1);
    check(wrong_slot == 0, "a pseudo p-value exactly where the CPU computes one");
    check_tie_bounds(vc.name, bounds, p_cpu, p_gpu, false);
}

int main(int argc, char* argv[])
{
    const char* lisa_path = (argc > 1) ? argv[1] : "Algorithms/lisa_kernel.metal";
    const char* jc_path = (argc > 2) ? argv[2] : "Algorithms/localjc_kernel.metal";

    if (!is_metal_supported()) {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("[SKIP] No Apple Silicon Metal device available (default device: %s, Apple7 family: %d).\n",
               dev ? [[dev name] UTF8String] : "none", dev ? (int)[dev supportsFamily:MTLGPUFamilyApple7] : 0);
        return 0;
    }
    TestCase g = sample_data("Guerry donatns, queen", guerry_n, guerry_x, guerry_nbr_offset, guerry_nbrs);
    TestCase u = sample_data("US Homicides hr90, queen", natregimes_n, natregimes_x, natregimes_nbr_offset, natregimes_nbrs);
    TestCase l = lattice(50);
    run(g, lisa_path, jc_path, 999);
    run(g, lisa_path, jc_path, 99999);
    run(u, lisa_path, jc_path, 999);
    run(l, lisa_path, jc_path, 999);

    TestCase d = dense(200, 100);
    run(d, lisa_path, jc_path, 999);
    TestCase td = tied_dense(1200, 150);
    run(td, lisa_path, jc_path, 999);
    TestCase ex = exact_ties(600);
    run(ex, lisa_path, jc_path, 999);

    run_jc_undef(l, jc_path, 999);
    run_jc_undef(u, jc_path, 999);

    // the other Local Moran variants on every test case
    const uint64_t seed = 123456789;
    const TestCase* cases[] = { &g, &u, &l, &d, &td };
    const char* case_names[] = { "Guerry", "US Homicides", "lattice", "dense", "tied values" };
    const struct {
        const char* name; int num_time_vals; bool bivariate, median; int undef_every;
    } variants[] = {
        { "bivariate Local Moran",                   1, true,  false, 0  },
        { "Local Moran, undefined values",           1, false, false, 11 },
        { "Local Moran, 3 time periods",             3, false, false, 0  },
        { "bivariate Local Moran, 3 periods, undefined values", 3, true, false, 7 },
        { "median Local Moran",                      1, false, true,  0  },
        { "median Local Moran, undefined values",    1, false, true,  11 },
    };
    for (size_t c = 0; c < sizeof(cases) / sizeof(*cases); c++) {
        for (size_t v = 0; v < sizeof(variants) / sizeof(*variants); v++) {
            VariantCase vc = build_variant(*cases[c], std::string(case_names[c]) + ": " + variants[v].name,
                                           variants[v].num_time_vals, variants[v].bivariate,
                                           variants[v].median, variants[v].undef_every, 999, seed);
            run_variant(vc, lisa_path);
        }
    }

    // an isolate leaves too few observations to draw 3 neighbors from: the CPU code
    // would never finish, the GPU code must refuse
    std::vector<GalElement> w(4);
    w[0].SetSizeNbrs(3);
    for (int j = 0; j < 3; j++) w[0].SetNbr(j, j + 1);
    w[1].SetSizeNbrs(1); w[1].SetNbr(0, 0);
    w[2].SetSizeNbrs(1); w[2].SetNbr(0, 0);
    double x4[4] = {1, -1, 2, 0}, lm4[4] = {0.3, -1, 2, 0}, p4[4];
    printf("too few observations to permute\n");
    check(!metal_lisa(lisa_path, 4, 99, 1, x4, lm4, w.data(), p4), "metal_lisa() returns false");

    printf(failures ? "%d CHECK(S) FAILED\n" : "ALL METAL TESTS PASSED\n", failures);
    return failures ? 1 : 0;
}
