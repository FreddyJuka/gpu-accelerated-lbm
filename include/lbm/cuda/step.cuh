#pragma once

// Host and device code. Included from .cu files only.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "lbm/config.hpp"
#include "lbm/cuda/check.hpp"
#include "lbm/grid.hpp"
#include "lbm/propagation/ab_pull.hpp"

namespace lbm {

// One thread per node: x along blockIdx.x * blockDim.x + threadIdx.x, y from
// blockIdx.y. Threads past the end of a row return. Grid2D and the collision
// policy are passed by value.
template <typename L, typename T, typename Collision>
__global__ void step_kernel(Grid2D<L> g, const NodeType* flags, const T* src, T* dst,
                            Collision coll) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y);
    if (x >= g.nx()) return;
    pull_stream_collide<L, T>(g, flags, src, dst, coll, x, y);
}

// Launches one pull stream-collide step with halfway bounce-back from d_src to
// d_dst (device pointers; d_flags holds one NodeType per node). Asynchronous:
// no device synchronization is performed. Throws
// std::invalid_argument for null pointers (including d_flags), d_src == d_dst, or ny above the
// current device's maximum grid y dimension (65535 on current architectures;
// the grid's y dimension is ny), and std::runtime_error if the launch fails.
template <typename L, typename T, typename Collision>
void step_gpu(const Grid2D<L>& g, const NodeType* d_flags, const T* d_src, T* d_dst,
              const Collision& coll) {
    if (d_flags == nullptr) throw std::invalid_argument("step_gpu: flags is null");
    if (d_src == nullptr) throw std::invalid_argument("step_gpu: src is null");
    if (d_dst == nullptr) throw std::invalid_argument("step_gpu: dst is null");
    if (d_src == d_dst) throw std::invalid_argument("step_gpu: src and dst must differ");

    int device = 0;
    LBM_CUDA_CHECK(cudaGetDevice(&device));
    int max_grid_y = 0;
    LBM_CUDA_CHECK(cudaDeviceGetAttribute(&max_grid_y, cudaDevAttrMaxGridDimY, device));
    if (g.ny() > max_grid_y)
        throw std::invalid_argument("step_gpu: ny = " + std::to_string(g.ny()) +
                                    " exceeds the device maximum grid y dimension " +
                                    std::to_string(max_grid_y));

    constexpr int block_x = 128;
    const dim3 block(block_x, 1, 1);
    const dim3 grid(static_cast<unsigned>((g.nx() + block_x - 1) / block_x),
                    static_cast<unsigned>(g.ny()), 1);
    step_kernel<L, T, Collision><<<grid, block>>>(g, d_flags, d_src, d_dst, coll);
    LBM_CUDA_CHECK(cudaGetLastError());
}

} // namespace lbm
