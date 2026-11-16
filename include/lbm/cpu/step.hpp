#pragma once

#include <stdexcept>

#include "lbm/grid.hpp"
#include "lbm/propagation/ab_pull.hpp"

namespace lbm {

// Host-only CPU reference step: one fused pull stream-collide pass with
// halfway bounce-back at wall nodes (flags, one NodeType per node) from src to
// dst. Wall nodes are not updated. Throws std::invalid_argument if flags, src
// or dst is null or if src == dst (in-place stepping is invalid for the
// two-buffer pull scheme: a node would overwrite populations its neighbours
// still have to read).
//
// src and dst hold Storage values; the collision runs in the policy's compute
// type (see pull_stream_collide and lbm/storage.hpp).
template <typename L, typename Storage, typename Collision>
void step_cpu(const Grid2D<L>& g, const NodeType* flags, const Storage* src, Storage* dst,
              const Collision& coll) {
    if (flags == nullptr) throw std::invalid_argument("step_cpu: flags is null");
    if (src == nullptr) throw std::invalid_argument("step_cpu: src is null");
    if (dst == nullptr) throw std::invalid_argument("step_cpu: dst is null");
    if (src == dst) throw std::invalid_argument("step_cpu: src and dst must differ");

    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x)
            pull_stream_collide<L, Storage>(g, flags, src, dst, coll, x, y);
}

} // namespace lbm
