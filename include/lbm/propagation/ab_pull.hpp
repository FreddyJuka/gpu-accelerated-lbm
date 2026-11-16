#pragma once

#include "lbm/config.hpp"
#include "lbm/grid.hpp"
#include "lbm/storage.hpp"

namespace lbm {

// Fused pull stream-collide with halfway bounce-back for one node (x, y),
// reading from src and writing to dst (two distinct buffers, SoA layout of
// Grid2D). src holds post-collision populations.
//
//   wall node:  return, dst is left untouched.
//   gather:     (xs, ys) = (wrap(x - c_qx), wrap(y - c_qy))
//                 s_q = src[idx(opp(q), x, y)]   if flags[node(xs, ys)] is Wall
//                 s_q = src[idx(q, xs, ys)]      otherwise
//   load:       f_q = load(s_q)
//   collide:    coll.collide<L>(f)
//   store:      dst[idx(q, x, y)] = store(f_q).
//
// The buffers hold Storage values; the collision runs in
// Compute = collision_compute_t<Collision, Storage> (the policy's
// compute_type, or Storage for a policy that does not declare one). load and
// store are those of StorageConversion<Storage, Compute> (lbm/storage.hpp);
// with Storage == Compute they are the identity. The gather, and with it the
// bounce-back, moves Storage values only.
//
// The domain is periodic wherever there are no walls. Neighbour flags are read
// at runtime. Precondition: 0 <= x < nx, 0 <= y < ny (the wrap handles
// single-step neighbours only). This is the single node implementation shared
// by the CPU reference and the CUDA kernel.
template <typename L, typename Storage, typename Collision>
LBM_HD inline void pull_stream_collide(const Grid2D<L>& g, const NodeType* flags,
                                       const Storage* src, Storage* dst, const Collision& coll,
                                       int x, int y) {
    using Compute = collision_compute_t<Collision, Storage>;
    using Conversion = StorageConversion<Storage, Compute>;
    static_assert(Conversion::supported,
                  "lbm: unsupported Storage/Compute combination: no StorageConversion<Storage, "
                  "Compute> specialization exists for the buffer type (Storage) and the "
                  "collision policy's compute_type (Compute); see lbm/storage.hpp");

    if (flags[g.node(x, y)] == NodeType::Wall) return;

    Compute f[L::Q];
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) {
        const int xs = g.wrap_x(x - L::c(q, 0));
        const int ys = g.wrap_y(y - L::c(q, 1));
        const Storage s = (flags[g.node(xs, ys)] == NodeType::Wall) ? src[g.idx(L::opp(q), x, y)]
                                                                    : src[g.idx(q, xs, ys)];
        f[q] = Conversion::load(s);
    }

    coll.template collide<L>(f);

    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) dst[g.idx(q, x, y)] = Conversion::store(f[q]);
}

} // namespace lbm
