#pragma once

#include <stdexcept>

#include "lbm/grid.hpp"
#include "lbm/propagation/ab_pull.hpp"

namespace lbm {

// Host-only CPU reference step on a periodic domain: one fused pull
// stream-collide pass from src to dst. Throws std::invalid_argument if either
// pointer is null or if src == dst (in-place stepping is invalid for the
// two-buffer pull scheme: a node would overwrite populations its neighbours
// still have to read).
template <typename L, typename T, typename Collision>
void step_periodic_cpu(const Grid2D<L>& g, const T* src, T* dst, const Collision& coll) {
    if (src == nullptr) throw std::invalid_argument("step_periodic_cpu: src is null");
    if (dst == nullptr) throw std::invalid_argument("step_periodic_cpu: dst is null");
    if (src == dst) throw std::invalid_argument("step_periodic_cpu: src and dst must differ");

    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x) pull_stream_collide_periodic<L, T>(g, src, dst, coll, x, y);
}

} // namespace lbm
