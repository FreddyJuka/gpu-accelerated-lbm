// Compile-only check: the lattice, grid, equilibrium, moments, TRT (unforced and forced) and pull step code must
// compile in device code for float and double without warnings. The kernels
// are never launched; results are written to global memory so that the code
// is not optimized away and the generated SASS can be inspected.

#include "lbm/collision/trt.hpp"
#include "lbm/config.hpp"
#include "lbm/equilibrium.hpp"
#include "lbm/force.hpp"
#include "lbm/grid.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "lbm/moments.hpp"
#include "lbm/propagation/ab_pull.hpp"

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

// Forced TRT (Guo scheme with uniform force); the policy is passed by value as
// part of the collision object.
template <typename T>
__global__ void check_trt_forced(lbm::TRT<T, lbm::UniformForce<T, L::D>> trt, T* f) {
    const int i = static_cast<int>(threadIdx.x);
    T floc[L::Q];
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) floc[q] = f[L::Q * i + q];
    trt.template collide<L>(floc);
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) f[L::Q * i + q] = floc[q];
}

// Forced pull step: node function with the forced TRT.
template <typename T>
__global__ void check_pull_step_forced(lbm::Grid2D<L> grid, const lbm::NodeType* flags,
                                       lbm::TRT<T, lbm::UniformForce<T, L::D>> trt, const T* src,
                                       T* dst) {
    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    lbm::pull_stream_collide<L, T>(grid, flags, src, dst, trt, x, y);
}

// Grid is passed by value, as in the production kernels. Reads one population
// from the periodic neighbour (pull scheme) of node (x, y).
template <typename T>
__global__ void check_grid(lbm::Grid2D<L> grid, const T* f, T* out) {
    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    T acc = T(0);
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) {
        const int xs = grid.wrap_x(x - L::c(q, 0));
        const int ys = grid.wrap_y(y - L::c(q, 1));
        acc += f[grid.idx(q, xs, ys)];
    }
    out[grid.node(x, y)] = acc;
}

// One thread per node; Grid2D and TRT are passed by value.
template <typename T>
__global__ void check_pull_step(lbm::Grid2D<L> grid, const lbm::NodeType* flags, lbm::TRT<T> trt,
                                const T* src, T* dst) {
    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    lbm::pull_stream_collide<L, T>(grid, flags, src, dst, trt, x, y);
}

template __global__ void check_equilibrium<float>(const float*, const float*, float*);
template __global__ void check_equilibrium<double>(const double*, const double*, double*);
template __global__ void check_moments<float>(const float*, float*, float*);
template __global__ void check_moments<double>(const double*, double*, double*);
template __global__ void check_trt<float>(lbm::TRT<float>, float*, float*);
template __global__ void check_trt<double>(lbm::TRT<double>, double*, double*);
template __global__ void check_pull_step<float>(lbm::Grid2D<L>, const lbm::NodeType*, lbm::TRT<float>,
                                                const float*, float*);
template __global__ void check_pull_step<double>(lbm::Grid2D<L>, const lbm::NodeType*, lbm::TRT<double>,
                                                 const double*, double*);
template __global__ void check_trt_forced<float>(lbm::TRT<float, lbm::UniformForce<float, 2>>, float*);
template __global__ void check_trt_forced<double>(lbm::TRT<double, lbm::UniformForce<double, 2>>,
                                                  double*);
template __global__ void check_pull_step_forced<float>(lbm::Grid2D<L>, const lbm::NodeType*,
                                                       lbm::TRT<float, lbm::UniformForce<float, 2>>,
                                                       const float*, float*);
template __global__ void check_pull_step_forced<double>(
    lbm::Grid2D<L>, const lbm::NodeType*, lbm::TRT<double, lbm::UniformForce<double, 2>>,
    const double*, double*);
template __global__ void check_grid<float>(lbm::Grid2D<L>, const float*, float*);
template __global__ void check_grid<double>(lbm::Grid2D<L>, const double*, double*);

} // namespace
