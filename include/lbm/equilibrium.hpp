#pragma once

#include "lbm/config.hpp"

namespace lbm {

// Shifted second-order equilibrium distribution
//
//   f~eq_q = feq_q - w_q
//          = w_q * ( drho + rho * ( cu/cs2 + cu^2/(2 cs2^2) - uu/(2 cs2) ) ),
//
// with rho = 1 + drho, cu = c_q . u, uu = u . u. The expression is evaluated
// directly in terms of the density deviation drho; feq_q is never formed, so
// no cancellation against w_q occurs. It is written with inv_cs2 = 1/cs2 so
// that no division is performed (for cs2 = 1/3: 3 cu + 4.5 cu^2 - 1.5 uu, all
// constants exact in float and double). T is the compute type.
template <typename L, typename T>
LBM_HD inline T equilibrium_shifted(int q, T drho, const T (&u)[L::D]) {
    const T inv_cs2 = L::template inv_cs2<T>();
    const T rho = T(1) + drho;

    T cu = T(0);
    LBM_UNROLL
    for (int d = 0; d < L::D; ++d) cu += T(L::c(q, d)) * u[d];

    T uu = T(0);
    LBM_UNROLL
    for (int d = 0; d < L::D; ++d) uu += u[d] * u[d];

    const T poly = inv_cs2 * cu + T(0.5) * inv_cs2 * inv_cs2 * cu * cu - T(0.5) * inv_cs2 * uu;
    return L::template w<T>(q) * (drho + rho * poly);
}

} // namespace lbm
