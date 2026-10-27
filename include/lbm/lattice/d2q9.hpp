#pragma once

#include "lbm/config.hpp"

namespace lbm {

// D2Q9 velocity set.
//
// Direction ordering:
//   0: ( 0, 0)   rest
//   1: ( 1, 0)   2: ( 0, 1)   3: (-1, 0)   4: ( 0,-1)
//   5: ( 1, 1)   6: (-1, 1)   7: (-1,-1)   8: ( 1,-1)
//
// Constants live in local constexpr arrays inside LBM_HD functions because
// nvcc rejects static constexpr member arrays in device code.
struct D2Q9 {
    static constexpr int D = 2;
    static constexpr int Q = 9;

    LBM_HD static constexpr int cx(int q) {
        constexpr int v[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
        return v[q];
    }

    LBM_HD static constexpr int cy(int q) {
        constexpr int v[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
        return v[q];
    }

    // Generic component accessor (d = 0: x, d = 1: y).
    LBM_HD static constexpr int c(int q, int d) {
        return d == 0 ? cx(q) : cy(q);
    }

    // Index of the direction opposite to q.
    LBM_HD static constexpr int opp(int q) {
        constexpr int v[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
        return v[q];
    }

    // Weights, computed in the target precision T.
    template <typename T>
    LBM_HD static constexpr T w(int q) {
        constexpr T v[Q] = {T(4) / T(9),
                            T(1) / T(9),  T(1) / T(9),  T(1) / T(9),  T(1) / T(9),
                            T(1) / T(36), T(1) / T(36), T(1) / T(36), T(1) / T(36)};
        return v[q];
    }

    // Lattice speed of sound squared.
    template <typename T>
    LBM_HD static constexpr T cs2() {
        return T(1) / T(3);
    }
};

} // namespace lbm
