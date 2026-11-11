#pragma once

// Rounding-error bounds shared by the numerical tests: the unit roundoff
// helpers, every operation-count constant K_* with its derivation, and the
// long double scales and references the bounds are expressed in.
//
// Conventions (see CLAUDE.md, "Numerical verification rules"):
//  - u = eps/2 is the unit roundoff of the type in question. Every rounding is
//    a relative error <= u of the quantity it produces.
//  - A constant K_x is an upper bound on a counted number of roundings. It is
//    written in units of eps (K_sum, K_u, K_e) or of u (all others); the unit is
//    stated at each definition. None is fitted to data.
//  - Scales are sums of the magnitudes of the terms involved, never the
//    magnitude of a result that may cancel.
//  - Every constant is <= 16 (static_assert at the end of the constants).
//  - Each constant has exactly one name, one meaning and one derivation, here.

#include <cmath>
#include <limits>

#include "support/fields.hpp"

namespace lbmtest {

using ld = long double;

// The references are evaluated in long double; they must be more precise than
// double, or they cannot resolve the rounding error of the double path.
static_assert(std::numeric_limits<ld>::digits >= 64,
              "long double must be extended precision for the reference");

// ---------------------------------------------------------------------------
// Epsilon and unit roundoff
// ---------------------------------------------------------------------------
template <typename T>
inline constexpr ld kEps = static_cast<ld>(std::numeric_limits<T>::epsilon());
template <typename T>
inline constexpr ld kU = 0.5L * kEps<T>;
inline constexpr ld kEpsLd = kEps<ld>;
inline constexpr ld kULd = kU<ld>;

// ---------------------------------------------------------------------------
// Operation-count constants
// ---------------------------------------------------------------------------

// Sequential sum of Q terms whose products with the integer velocity components
// c in {-1, 0, 1} are exact (moments_shifted, moments_shifted_forced, the
// lattice moment tests): recursive summation satisfies
//   |err| <= (Q - 1) u sum|term| (1 + O(Q u)).
// K_acc is that count in units of u, for tests that account for the rounding
// of the sum themselves (and then also state their headroom).
inline constexpr ld K_acc = L::Q - 1;
// K_sum is the same bound in units of eps with 1/2 eps of headroom for the
// O(Q u) second-order factor: K_sum eps = ((Q - 1)/2 + 1/2) eps. The actual
// second-order contribution is ~Q^2 u^2, many orders below it.
inline constexpr ld K_sum = (K_acc + 1.0L) / 2.0L;

// Division and product u_d = j_d * inv_rho of moments_shifted, in units of eps.
// Let drho~ = drho + delta and j~ = j + eps_j with |delta|, |eps_j| <= E
// (E = K_sum eps S), rho = 1 + drho > 0:
//   rho~     = fl(1 + drho~) : relative error vs rho  <= |delta|/rho + u
//   inv_rho~ = fl(1/rho~)    : relative error vs 1/rho <= |delta|/rho + 2u
//   u~       = fl(j~ inv_rho~): relative error adds u, so vs the exact j/rho
//   |u~ - u| <= |eps_j|/rho + |u| (|delta|/rho + 3u)    (first order)
//            <= E (1 + |u|)/rho + 1.5 eps |u|.
// K_u = 2 adds 1/2 eps of headroom for the second-order terms (relative size
// ~(E/rho)^2). tol_u = E (1 + |u|)/rho + K_u eps |u|, see tol_u_from.
inline constexpr ld K_u = 2.0L;

// Velocity of the forced moments, u = (j + F/2) * inv_rho (moments_shifted_forced),
// in units of u: the addition, the division and the product are 3 roundings
// (F/2 is exact).
inline constexpr ld K_div = 3.0L;

// Shifted equilibrium f~eq_q (equilibrium_shifted), in units of eps. Counting
// the roundings of every term with u = eps/2:
//   cu = c.u        : the products are exact (c in {-1, 0, 1}); one addition  -> 1
//   t1 = a cu       : cu (1) + product (1)                                    -> 2
//   t2 = 0.5 a a cu cu, evaluated ((0.5 a) a) cu) cu: the constants are exact,
//                     cu appears twice (1 each) and there are 2 products      -> 4
//   t3 = 0.5 a uu   : uu = 2 products + 1 addition (2, relative since all
//                     terms are >= 0) + 1 product                             -> 3
//   poly = (t1 + t2) - t3: two additions; t1 and t2 pass through both, t3 only
//                     through the second: t1: 2+2 = 4, t2: 4+2 = 6, t3: 3+1 = 4,
//                     so |err(poly)| <= 6u (|t1| + |t2| + |t3|)
//   rho = 1 + drho (1) and rho*poly (1): 8u on rho (|t1| + |t2| + |t3|)
//   drho + rho*poly (1 addition), w*(...) (1 product), and the rounding of
//   w<T> itself (1): 3 more, 11u on the sum of magnitudes (the |drho| term
//   only gets 3u, which 11u covers).
// Hence |err| <= 11u scale_e = 5.5 eps scale_e. K_e = 6 leaves 0.5 eps = 1u of
// headroom for the O(n u) second-order terms of (1 + u)^n - 1 (about 66 u^2 for
// n = 11); it is an upper bound, not a fitted value.
inline constexpr ld K_e = 6.0L;

// Collision chain of out_q in the forced TRT, in units of u: fneq_q, fneq_opp
// (2), a + b and a - b (2), omega * fneq_pm (2), two subtractions (2),
// (1 - omega/2) twice (2), two source products (2), two additions (2) = 14
// roundings, each of relative size u on a quantity bounded by
// M_q + N_q + N_opp (defined at the use).
inline constexpr ld K_chain = 14.0L;

// guo_source_parts, D = 2, in units of u: s_anti = (w a) cF has 3 roundings
// (cF: 1; w a: 1; product: 1); s_sym has 10: cu 1, cF 1, (a a) cu 1, (.) cF 1,
// uF 3 (two products, one sum), a uF 1, difference 1, w * (.) 1 (a a is exact:
// a = 3). Each rounding changes s_q by at most u times the magnitude of the
// terms it affects, so |s_q - exact| <= K_src u W_q with W_q = source_scale.
// 10 is the larger count and also covers s_anti.
inline constexpr ld K_src = 10.0L;

// Representation of the weights w_q(T) = T(4)/T(9) etc., in units of u: a
// relative error of at most u, so an identity that uses sum_q w_q ... holds up
// to u times the sum of magnitudes.
inline constexpr ld K_w = 1.0L;

// Reference right-hand side u_a F_b + u_b F_a, in units of u: 2 products and
// 1 sum = 3 roundings.
inline constexpr ld K_uF = 3.0L;

// Long double reference of the shifted equilibrium, f~eq_ref = feq_classical_ld
// - w<ld>(q), in units of eps_ld; counted with u_ld = eps_ld/2 against
// ref_scale:
//   rho = 1 + drho                       -> 1
//   cs2<ld> = 1/3 (rounded)              -> 1 wherever cs2 appears
//   cu : one addition (c*u exact)        -> 1;   uu : 2 products + 1 add -> 2
//   t1 = cu/cs2             : cu 1 + cs2 1 + division 1               -> 3
//   t2 = cu*cu/(2 cs2 cs2)  : numerator 1+1+1, denominator 1+1+1
//                             (2*cs2 is exact), division 1             -> 7
//   t3 = uu/(2 cs2)         : uu 2 + cs2 1 + division 1                -> 4
//   S = ((1 + t1) + t2) - t3: three additions; 1 and t1 pass through all
//   three, t2 through two, t3 through one:
//   1: 3, t1: 3+3 = 6, t2: 7+2 = 9, t3: 4+1 = 5 -> |err(S)| <= 9u_ld sum|terms|.
//   (w*rho)*S : w<ld> 1, rho 1, product w*rho 1, final product 1
//   -> 9 + 4 = 13u_ld on ref_scale.
//   The subtraction feq - w is exact by Sterbenz's lemma (asserted where it is
//   used), but the w<ld>(q) it subtracts is rounded: the exact shifted value
//   is feq - w_exact, so the reference carries one more independent error
//   u_ld w. With w <= ref_scale/rho and rho >= kRhoMin = 0.98 (asserted where
//   it is used) that is <= 1.021 u_ld ref_scale.
// Total 14.03 u_ld = 7.02 eps_ld; K_feq_ld = 8 leaves 0.98 eps_ld of headroom
// for the second-order terms of (1 + u)^n - 1 (about 100 u_ld^2).
inline constexpr ld K_feq_ld = 8.0L;

static_assert(K_acc <= 16.0L && K_sum <= 16.0L && K_u <= 16.0L && K_div <= 16.0L &&
                  K_e <= 16.0L && K_chain <= 16.0L && K_src <= 16.0L && K_w <= 16.0L &&
                  K_uF <= 16.0L && K_feq_ld <= 16.0L,
              "every factor must not exceed 16");
static_assert(K_sum == L::Q / 2.0L, "K_sum is (Q - 1)/2 + 1/2 in units of eps");

// ---------------------------------------------------------------------------
// Scales and references (long double)
// ---------------------------------------------------------------------------

// Classical unshifted equilibrium in long double.
inline ld feq_classical_ld(int q, ld drho, const ld (&u)[2]) {
    const ld rho = 1.0L + drho;
    const ld cs2 = L::cs2<ld>();
    const ld cu = L::c(q, 0) * u[0] + L::c(q, 1) * u[1];
    const ld uu = u[0] * u[0] + u[1] * u[1];
    return L::w<ld>(q) * rho * (1.0L + cu / cs2 + cu * cu / (2.0L * cs2 * cs2) - uu / (2.0L * cs2));
}

// Sum of magnitudes of the terms of feq_classical_ld (the scale of its
// rounding error, K_feq_ld ref_scale).
inline ld ref_scale(int q, ld drho, const ld (&u)[2]) {
    const ld rho = 1.0L + drho;
    const ld cs2 = L::cs2<ld>();
    const ld cu = L::c(q, 0) * u[0] + L::c(q, 1) * u[1];
    const ld uu = u[0] * u[0] + u[1] * u[1];
    return L::w<ld>(q) * rho *
           (1.0L + std::abs(cu) / cs2 + cu * cu / (2.0L * cs2 * cs2) + uu / (2.0L * cs2));
}

// Sum of the magnitudes of the terms of f~eq_q (equilibrium_shifted), the scale
// of its rounding error |err| <= K_e eps scale_e, evaluated in long double at
// inputs already rounded to the compute type T.
template <typename T>
ld scale_e(int q, T drho_t, const T (&u_t)[L::D]) {
    const ld drho = drho_t;
    const ld rho = 1.0L + drho;
    const ld a = L::inv_cs2<ld>();
    ld cu = 0.0L, uu = 0.0L;
    for (int d = 0; d < L::D; ++d) {
        cu += L::c(q, d) * static_cast<ld>(u_t[d]);
        uu += static_cast<ld>(u_t[d]) * static_cast<ld>(u_t[d]);
    }
    return L::w<ld>(q) *
           (std::abs(drho) + rho * (a * std::abs(cu) + 0.5L * a * a * cu * cu + 0.5L * a * uu));
}

// Sum of the magnitudes of the terms of s_anti + s_sym at direction q
// (guo_source_parts), the scale W_q of K_src.
template <typename T>
ld source_scale(int q, const T (&u)[L::D], const T (&F)[L::D]) {
    const ld a = L::inv_cs2<ld>();
    ld cu = 0.0L, cF = 0.0L, uF = 0.0L;
    for (int d = 0; d < L::D; ++d) {
        cu += std::abs(static_cast<ld>(L::c(q, d))) * std::abs(static_cast<ld>(u[d]));
        cF += std::abs(static_cast<ld>(L::c(q, d))) * std::abs(static_cast<ld>(F[d]));
        uF += std::abs(static_cast<ld>(u[d])) * std::abs(static_cast<ld>(F[d]));
    }
    return L::w<ld>(q) * (a * a * cu * cF + a * uF + a * cF);
}

// Division and product u_d = j_d * inv_rho (see K_u):
//   |u~ - u| <= E (1 + |u|)/rho + K_u eps |u|,  E = absolute error of drho, j.
inline ld tol_u_from(ld E, ld eps, ld rho, ld absu) {
    return E * (1.0L + absu) / rho + K_u * eps * absu;
}

}  // namespace lbmtest
