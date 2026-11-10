#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "lbm/config.hpp"

namespace lbm {

// Force policy of a collision operator without body force.
struct NoForce {};

// Uniform, constant body force F (force per unit volume, lattice units) with D
// components stored in the compute type T.
template <typename T, int D>
class UniformForce {
public:
    // Host-only validating factory. Throws std::invalid_argument naming the
    // component and its value if a component is not finite or its magnitude
    // exceeds the largest finite T (e.g. a double beyond the float range).
    static UniformForce create(const double (&components)[D]) {
        UniformForce force;
        for (int d = 0; d < D; ++d) {
            // The range check precedes the cast: converting a double outside the
            // range of T is undefined behaviour. NaN fails isfinite first.
            if (!std::isfinite(components[d]) ||
                std::fabs(components[d]) > static_cast<double>(std::numeric_limits<T>::max()))
                throw std::invalid_argument("UniformForce: component F[" + std::to_string(d) +
                                            "] = " + std::to_string(components[d]) +
                                            " must be finite");
            force.F_[d] = static_cast<T>(components[d]);
        }
        return force;
    }

    LBM_HD constexpr T F(int d) const { return F_[d]; }

private:
    UniformForce() = default;

    T F_[D];
};

static_assert(std::is_trivially_copyable_v<UniformForce<float, 2>>);
static_assert(std::is_trivially_copyable_v<UniformForce<double, 2>>);
static_assert(std::is_trivially_copyable_v<UniformForce<float, 3>>);
static_assert(std::is_trivially_copyable_v<UniformForce<double, 3>>);

// Guo et al. (2002) forcing term, split into the part odd and the part even
// under q -> opp(q) so that TRT can relax each with its own rate:
//
//   S_q   = w_q [ a (c_q - u) + a^2 (c_q . u) c_q ] . F,       a = 1/cs2
//   s_anti = w_q a (c_q . F)                                   (odd)
//   s_sym  = w_q [ a^2 (c_q . u)(c_q . F) - a (u . F) ]        (even)
//   S_q   = s_anti + s_sym.
//
// With c_q . F = cF, c_q . u = cu, u . F = uF. No divisions. u is the Guo
// half-force velocity (see moments_shifted_forced). T is the compute type.
template <typename L, typename T>
LBM_HD inline void guo_source_parts(int q, const T (&u)[L::D], const T (&F)[L::D], T& s_anti,
                                    T& s_sym) {
    const T a = L::template inv_cs2<T>();

    T cF = T(0);
    LBM_UNROLL
    for (int d = 0; d < L::D; ++d) cF += T(L::c(q, d)) * F[d];

    T cu = T(0);
    LBM_UNROLL
    for (int d = 0; d < L::D; ++d) cu += T(L::c(q, d)) * u[d];

    T uF = T(0);
    LBM_UNROLL
    for (int d = 0; d < L::D; ++d) uF += u[d] * F[d];

    const T w = L::template w<T>(q);
    s_anti = w * a * cF;
    s_sym = w * (a * a * cu * cF - a * uF);
}

} // namespace lbm
