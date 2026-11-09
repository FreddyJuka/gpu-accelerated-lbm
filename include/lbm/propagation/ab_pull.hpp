#pragma once

#include "lbm/config.hpp"
#include "lbm/grid.hpp"

namespace lbm {

// Fused pull stream-collide with halfway bounce-back for one node (x, y),
// reading from src and writing to dst (two distinct buffers, SoA layout of
// Grid2D). src holds post-collision populations.
//
//   wall node:  return, dst is left untouched.
//   gather:     (xs, ys) = (wrap(x - c_qx), wrap(y - c_qy))
//                 f_q = src[idx(opp(q), x, y)]   if flags[node(xs, ys)] is Wall
//                 f_q = src[idx(q, xs, ys)]      otherwise
//   collide:    coll.collide<L>(f)
//   store:      dst[idx(q, x, y)] = f_q.
//
// The domain is periodic wherever there are no walls. Neighbour flags are read
// at runtime. Precondition: 0 <= x < nx, 0 <= y < ny (the wrap handles
// single-step neighbours only). This is the single node implementation shared
// by the CPU reference and the CUDA kernel.
template <typename L, typename T, typename Collision>
LBM_HD inline void pull_stream_collide(const Grid2D<L>& g, const NodeType* flags, const T* src,
                                       T* dst, const Collision& coll, int x, int y) {
    if (flags[g.node(x, y)] == NodeType::Wall) return;

    T f[L::Q];
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) {
        const int xs = g.wrap_x(x - L::c(q, 0));
        const int ys = g.wrap_y(y - L::c(q, 1));
        f[q] = (flags[g.node(xs, ys)] == NodeType::Wall) ? src[g.idx(L::opp(q), x, y)]
                                                         : src[g.idx(q, xs, ys)];
    }

    coll.template collide<L>(f);

    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) dst[g.idx(q, x, y)] = f[q];
}

} // namespace lbm
