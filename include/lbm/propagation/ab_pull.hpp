#pragma once

#include "lbm/config.hpp"
#include "lbm/grid.hpp"

namespace lbm {

// Fused pull stream-collide for one node (x, y) of a periodic domain, reading
// from src and writing to dst (two distinct buffers, SoA layout of Grid2D):
//
//   gather:  f_q = src[idx(q, wrap(x - c_qx), wrap(y - c_qy))]
//   collide: coll.collide<L>(f)
//   store:   dst[idx(q, x, y)] = f_q.
//
// Precondition: 0 <= x < nx, 0 <= y < ny (the wrap handles single-step
// neighbours only). This is the single node implementation shared by the CPU
// reference and the CUDA kernel.
template <typename L, typename T, typename Collision>
LBM_HD inline void pull_stream_collide_periodic(const Grid2D<L>& g, const T* src, T* dst,
                                                const Collision& coll, int x, int y) {
    T f[L::Q];
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) {
        const int xs = g.wrap_x(x - L::c(q, 0));
        const int ys = g.wrap_y(y - L::c(q, 1));
        f[q] = src[g.idx(q, xs, ys)];
    }

    coll.template collide<L>(f);

    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) dst[g.idx(q, x, y)] = f[q];
}

} // namespace lbm
