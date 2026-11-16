#pragma once

// Host and device code. Included from .cu files only.

#include <cuda_runtime.h>

#include <stdexcept>

#include "lbm/config.hpp"
#include "lbm/cuda/check.hpp"
#include "lbm/grid.hpp"
#include "lbm/propagation/ab_pull.hpp"

namespace lbm {

// One thread per node over the flattened node index
// node = blockIdx.x * blockDim.x + threadIdx.x = y * nx + x (the SoA node
// order, x fastest). Threads past the last node return. Consecutive threads
// therefore handle consecutive nodes across row boundaries, so every warp is
// fully populated whatever nx is: a narrow grid (the nx = 4 validation
// channel) or an nx that is not a multiple of 32 no longer leaves lanes idle,
// which matters because the cost of a warp instruction (in particular on the
// FP64 pipe) does not depend on the number of active lanes. Measured on the
// RTX 3050 Laptop: double nx = 4 runs 7.5x faster, nx = 16 2x, nx = 100 1.3x,
// square grids unchanged. Coalescing is the same as with one row per block:
// the SoA index is q * N + node, contiguous in node. Grid2D and the collision
// policy are passed by value.
template <typename L, typename T, typename Collision>
__global__ void step_kernel(Grid2D<L> g, const NodeType* flags, const T* src, T* dst,
                            Collision coll) {
    const int node = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (node >= g.num_nodes()) return;
    const int y = node / g.nx();
    const int x = node - y * g.nx();
    pull_stream_collide<L, T>(g, flags, src, dst, coll, x, y);
}

// Launches one pull stream-collide step with halfway bounce-back from d_src to
// d_dst (device pointers; d_flags holds one NodeType per node). Asynchronous:
// no device synchronization is performed. Throws
// std::invalid_argument for null pointers (including d_flags) or d_src == d_dst,
// and std::runtime_error if the launch fails.
//
// The only size limit is the one of Grid2D, Q * nx * ny <= INT32_MAX. The grid
// is one-dimensional with ceil(N / 128) blocks, N = nx * ny <= INT32_MAX / Q,
// so at most ceil((2^31 - 1) / 128) = 2^24 blocks, far below the grid x limit
// of 2^31 - 1. The launch does not depend on ny.
template <typename L, typename T, typename Collision>
void step_gpu(const Grid2D<L>& g, const NodeType* d_flags, const T* d_src, T* d_dst,
              const Collision& coll) {
    if (d_flags == nullptr) throw std::invalid_argument("step_gpu: flags is null");
    if (d_src == nullptr) throw std::invalid_argument("step_gpu: src is null");
    if (d_dst == nullptr) throw std::invalid_argument("step_gpu: dst is null");
    if (d_src == d_dst) throw std::invalid_argument("step_gpu: src and dst must differ");

    // 128 threads per block: measured equal to 64 and 256 within the run-to-run
    // spread for every case; 128 keeps the 100 % theoretical occupancy of the
    // 40-register kernels (12 blocks of 128 = 1536 threads per SM on sm_86).
    constexpr int block_x = 128;
    const dim3 block(block_x, 1, 1);
    const dim3 grid(static_cast<unsigned>((g.num_nodes() + block_x - 1) / block_x), 1, 1);
    step_kernel<L, T, Collision><<<grid, block>>>(g, d_flags, d_src, d_dst, coll);
    LBM_CUDA_CHECK(cudaGetLastError());
}

} // namespace lbm
