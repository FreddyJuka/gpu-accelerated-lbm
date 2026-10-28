#pragma once

#include "lbm/config.hpp"

namespace lbm {

// Macroscopic moments from shifted populations f~_q = f_q - w_q
//
//   drho = sum_q f~_q
//   j_d  = sum_q f~_q c_qd            (= rho u_d exactly, since sum_q w_q c_q = 0)
//   u_d  = j_d / rho,  rho = 1 + drho.
//
// The density deviation is summed directly from the shifted populations; f_q is
// never formed, so no cancellation against w_q occurs. The factors c_qd are in
// {-1, 0, 1} and the products exact. One division per call (inv_rho); u_d is
// then a product. T is the compute type.
template <typename L, typename T>
LBM_HD inline void moments_shifted(const T (&f)[L::Q], T& drho, T (&u)[L::D]) {
    T sum = T(0);
    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) sum += f[q];

    T j[L::D];
    LBM_UNROLL
    for (int d = 0; d < L::D; ++d) j[d] = T(0);

    LBM_UNROLL
    for (int q = 0; q < L::Q; ++q) {
        LBM_UNROLL
        for (int d = 0; d < L::D; ++d) j[d] += T(L::c(q, d)) * f[q];
    }

    const T inv_rho = T(1) / (T(1) + sum);

    drho = sum;
    LBM_UNROLL
    for (int d = 0; d < L::D; ++d) u[d] = j[d] * inv_rho;
}

} // namespace lbm
