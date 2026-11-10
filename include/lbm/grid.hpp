#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "lbm/config.hpp"

namespace lbm {

// Node type flag, stored as one byte per node. More values will be added
// later; the underlying type stays uint8_t.
enum class NodeType : std::uint8_t { Fluid = 0, Wall = 1 };

// Structured 2D grid with Structure-of-Arrays indexing, x fastest:
//
//   node(x, y)   = y * nx + x
//   idx(q, x, y) = q * N + y * nx + x,   N = nx * ny.
//
// Indices are 32-bit int. The constructor verifies, without forming any
// product that can overflow, that Q * nx * ny <= INT32_MAX, so no index
// returned by idx() can overflow for 0 <= q < Q, 0 <= x < nx, 0 <= y < ny. The class is trivially copyable and
// is passed by value to kernels.
template <typename L>
class Grid2D {
    static_assert(L::D == 2, "Grid2D requires a two-dimensional lattice");

public:
    // Host-only. Throws std::invalid_argument if nx <= 0, ny <= 0 or
    // Q * nx * ny exceeds INT32_MAX.
    Grid2D(int nx, int ny) : nx_(nx), ny_(ny) {
        if (nx <= 0) throw std::invalid_argument("Grid2D: nx must be > 0, got " + std::to_string(nx));
        if (ny <= 0) throw std::invalid_argument("Grid2D: ny must be > 0, got " + std::to_string(ny));
        // Q*nx*ny > INT32_MAX is tested as nx > (INT32_MAX / Q) / ny, in integer
        // arithmetic, so no product is formed and nothing can overflow. For
        // positive integers it is exactly equivalent: Q*k <= M  <=>  k <= floor(M/Q),
        // and nx*ny <= K  <=>  nx <= floor(K/ny). Applying both with M = INT32_MAX,
        // K = floor(M/Q) gives Q*nx*ny <= M  <=>  nx <= floor(floor(M/Q)/ny).
        constexpr std::int64_t int_max = std::numeric_limits<std::int32_t>::max();
        constexpr std::int64_t max_nodes = int_max / L::Q;
        if (static_cast<std::int64_t>(nx) > max_nodes / ny) {
            // The message reports the product if it fits in int64_t. nx*ny < 2^62
            // always fits; Q times it may not, which is checked by division.
            const std::int64_t nodes = static_cast<std::int64_t>(nx) * ny;
            std::string got = "more than INT64_MAX";
            if (nodes <= std::numeric_limits<std::int64_t>::max() / L::Q)
                got = std::to_string(static_cast<std::int64_t>(L::Q) * nodes);
            throw std::invalid_argument("Grid2D: Q*nx*ny must be <= INT32_MAX, got " + got +
                                        " (Q = " + std::to_string(L::Q) +
                                        ", nx = " + std::to_string(nx) +
                                        ", ny = " + std::to_string(ny) + ")");
        }
    }

    LBM_HD constexpr int nx() const { return nx_; }
    LBM_HD constexpr int ny() const { return ny_; }

    // Number of nodes N = nx * ny.
    LBM_HD constexpr int num_nodes() const { return nx_ * ny_; }

    LBM_HD constexpr int node(int x, int y) const { return y * nx_ + x; }

    LBM_HD constexpr int idx(int q, int x, int y) const { return q * num_nodes() + y * nx_ + x; }

    // Periodic wrap for a single lattice step. Precondition: -1 <= x <= nx.
    // Conditional add/subtract instead of %, which is cheaper on the GPU.
    LBM_HD constexpr int wrap_x(int x) const { return x < 0 ? x + nx_ : (x >= nx_ ? x - nx_ : x); }

    // Periodic wrap for a single lattice step. Precondition: -1 <= y <= ny.
    LBM_HD constexpr int wrap_y(int y) const { return y < 0 ? y + ny_ : (y >= ny_ ? y - ny_ : y); }

private:
    int nx_, ny_;
};

} // namespace lbm
