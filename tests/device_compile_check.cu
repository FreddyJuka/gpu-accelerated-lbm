// Compile-only check: the lattice, equilibrium, moments and TRT code must
// compile in device code for float and double without warnings. The kernels
// are never launched; results are written to global memory so that the code
// is not optimized away and the generated SASS can be inspected.

#include "lbm/collision/trt.hpp"
#include "lbm/config.hpp"
#include "lbm/equilibrium.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "lbm/moments.hpp"

namespace {

using L = lbm::D2Q9;

template <typename T>
__global__ void check_equilibrium(const T* drho, const T* u, T* out) {
    const int i = static_cast<int>(threadIdx.x);
    T uloc[L::D];
    LBM_UNROLL
    for (int d = 0; d < L::D; ++d) uloc[d] = u[L::D * i + d];
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q)
        out[L::Q * i + q] = lbm::equilibrium_shifted<L, T>(q, drho[i], uloc);
}

template <typename T>
__global__ void check_moments(const T* f, T* drho, T* u) {
    const int i = static_cast<int>(threadIdx.x);
    T floc[L::Q];
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) floc[q] = f[L::Q * i + q];
    T r;
    T uloc[L::D];
    lbm::moments_shifted<L, T>(floc, r, uloc);
    drho[i] = r;
    LBM_UNROLL
    for (int d = 0; d < L::D; ++d) u[L::D * i + d] = uloc[d];
}

template <typename T>
__global__ void check_trt(lbm::TRT<T> trt, T* f, T* omegas) {
    const int i = static_cast<int>(threadIdx.x);
    T floc[L::Q];
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) floc[q] = f[L::Q * i + q];
    trt.template collide<L>(floc);
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) f[L::Q * i + q] = floc[q];
    omegas[2 * i] = trt.omega_plus();
    omegas[2 * i + 1] = trt.omega_minus();
}

template __global__ void check_equilibrium<float>(const float*, const float*, float*);
template __global__ void check_equilibrium<double>(const double*, const double*, double*);
template __global__ void check_moments<float>(const float*, float*, float*);
template __global__ void check_moments<double>(const double*, double*, double*);
template __global__ void check_trt<float>(lbm::TRT<float>, float*, float*);
template __global__ void check_trt<double>(lbm::TRT<double>, double*, double*);

} // namespace
