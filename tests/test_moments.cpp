#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "lbm/equilibrium.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "lbm/moments.hpp"

using L = lbm::D2Q9;

namespace {

struct State {
    double drho;
    double ux, uy;
};

// Same low-Mach state set as test_equilibrium.cpp.
std::vector<State> low_mach_states() {
    const double drhos[] = {0.0, 0.01, -0.02};
    const double us[][2] = {{0.0, 0.0}, {0.05, 0.0}, {0.0, -0.03}, {0.04, 0.07}};
    std::vector<State> s;
    for (double r : drhos)
        for (const auto& u : us) s.push_back({r, u[0], u[1]});
    return s;
}

// The reference must be more precise than double, or it cannot resolve the
// rounding error of the double-precision path.
static_assert(std::numeric_limits<long double>::digits >= 64,
              "long double must be extended precision for the reference");

// Classical unshifted equilibrium in long double (as in test_equilibrium.cpp).
long double feq_classical_ld(int q, long double drho, const long double (&u)[2]) {
    const long double rho = 1.0L + drho;
    const long double cs2 = L::cs2<long double>();
    const long double cu = L::c(q, 0) * u[0] + L::c(q, 1) * u[1];
    const long double uu = u[0] * u[0] + u[1] * u[1];
    return L::w<long double>(q) * rho *
           (1.0L + cu / cs2 + cu * cu / (2.0L * cs2 * cs2) - uu / (2.0L * cs2));
}

// Sum of magnitudes of the terms of feq_classical_ld (the scale of its rounding
// error), as ref_scale in test_equilibrium.cpp.
long double ref_scale(int q, long double drho, const long double (&u)[2]) {
    const long double rho = 1.0L + drho;
    const long double cs2 = L::cs2<long double>();
    const long double cu = L::c(q, 0) * u[0] + L::c(q, 1) * u[1];
    const long double uu = u[0] * u[0] + u[1] * u[1];
    return L::w<long double>(q) * rho *
           (1.0L + std::abs(cu) / cs2 + cu * cu / (2.0L * cs2 * cs2) + uu / (2.0L * cs2));
}

// Moments in long double from long-double populations, same formulas as the
// implementation (summation order q = 0..Q-1, one division).
void moments_ld(const long double (&f)[L::Q], long double& drho, long double (&u)[L::D]) {
    long double sum = 0.0L;
    long double j[L::D] = {0.0L, 0.0L};
    for (int q = 0; q < L::Q; ++q) {
        sum += f[q];
        for (int d = 0; d < L::D; ++d) j[d] += L::c(q, d) * f[q];
    }
    const long double inv_rho = 1.0L / (1.0L + sum);
    drho = sum;
    for (int d = 0; d < L::D; ++d) u[d] = j[d] * inv_rho;
}

// ---------------------------------------------------------------------------
// Error bounds. u = eps/2 is the unit roundoff of the type in question.
//
// Summation (drho and j_d). Recursive summation of n terms satisfies
//   |err| <= (n-1) u sum|term| (1 + O(n u)).
// Here n <= Q, and the c_qd in {-1,0,1} make every product exact, so for both
// drho and j_d the terms have magnitudes |f~_q c_qd| <= |f~_q| and
//   |err| <= (Q-1) u S = (Q-1)/2 eps S,   S = sum_q |f~_q|.
// K_sum = Q/2 adds 1/2 eps of headroom for the O(n u) second-order factor
// (actual second-order contribution is ~Q^2 u^2, many orders below it). It is
// an upper bound, not a fitted value. S is a sum of magnitudes, so it does not
// shrink when the populations cancel in drho or j.
//
// Division and product (u_d = j_d * inv_rho). Let drho~ = drho + delta and
// j~ = j + eps_j with |delta|, |eps_j| <= E (E = K_sum eps S in the pure
// arithmetic case). With rho = 1 + drho (> 0):
//   rho~     = fl(1 + drho~)  : relative error vs rho  <= |delta|/rho + u
//   inv_rho~ = fl(1/rho~)     : relative error vs 1/rho <= |delta|/rho + 2u
//   u~       = fl(j~ inv_rho~): relative error adds u, so vs the exact j/rho:
//   |u~ - u| <= |eps_j|/rho + |u| (|delta|/rho + 3u)     (to first order)
//            <= E (1 + |u|)/rho + 1.5 eps |u|.
// K_u = 2 adds 1/2 eps of headroom for the second-order terms (relative size
// ~(E/rho)^2, again many orders below). Hence
//   tol_u = E (1 + |u|)/rho + K_u eps |u|.
// ---------------------------------------------------------------------------
constexpr long double K_sum = L::Q / 2.0L;
constexpr long double K_u = 2.0L;
// Rounding of each input f~_q to T, at most u |f~_q| = 0.5 eps |f~_q|, enters
// as an additional absolute error of the terms of both sums: K_rnd = 1/2.
constexpr long double K_rnd = 0.5L;
// Error of feq_classical_ld, from test_equilibrium.cpp (13 u_ld on ref_scale,
// plus headroom).
constexpr long double K_ref = 7.0L;
static_assert(K_sum <= 16.0L && K_u <= 16.0L && K_rnd <= 16.0L && K_ref <= 16.0L,
              "every factor must not exceed 16");

long double tol_u_from(long double E, long double eps, long double rho, long double absu) {
    return E * (1.0L + absu) / rho + K_u * eps * absu;
}

template <typename T>
void inputs_from_state(const State& s, T& drho, T (&u)[2], T (&f)[L::Q]) {
    drho = static_cast<T>(s.drho);
    u[0] = static_cast<T>(s.ux);
    u[1] = static_cast<T>(s.uy);
    for (int q = 0; q < L::Q; ++q) f[q] = lbm::equilibrium_shifted<L, T>(q, drho, u);
}

} // namespace

template <typename T>
class MomentsShifted : public ::testing::Test {
protected:
    static T eps() { return std::numeric_limits<T>::epsilon(); }
};

using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MomentsShifted, PrecisionTypes);

// a. Rest state.
TYPED_TEST(MomentsShifted, RestStateIsExactlyZero) {
    using T = TypeParam;
    T f[L::Q];
    for (int q = 0; q < L::Q; ++q) f[q] = T(0);
    T drho = T(1), u[L::D] = {T(1), T(1)};
    lbm::moments_shifted<L, T>(f, drho, u);
    EXPECT_EQ(drho, T(0));
    for (int d = 0; d < L::D; ++d) EXPECT_EQ(u[d], T(0)) << "d = " << d;
}

// Check moments_shifted against the long double evaluation of the same formulas
// on the same T-valued populations. Only the moment arithmetic is tested.
// Tolerances (derivations above): the T-arithmetic error with eps = eps_T plus
// the long double reference's own arithmetic error with eps = eps_ld, each of
// the same form because the formulas are identical:
//   tol_drho = K_sum (eps_T + eps_ld) S
//   tol_u    = tol_u_from(K_sum eps_T S, eps_T, ...) +
//              tol_u_from(K_sum eps_ld S, eps_ld, ...)
// with S = sum_q |f~_q| of the T-valued inputs, rho and |u| from the reference.
template <typename T>
void check_arithmetic(const T (&f)[L::Q], const char* what, const State& s) {
    const long double eps_t = std::numeric_limits<T>::epsilon();
    const long double eps_ld = std::numeric_limits<long double>::epsilon();

    long double f_ld[L::Q];
    long double S = 0.0L;
    for (int q = 0; q < L::Q; ++q) {
        f_ld[q] = f[q];
        S += std::abs(f_ld[q]);
    }
    long double drho_ref, u_ref[L::D];
    moments_ld(f_ld, drho_ref, u_ref);
    const long double rho = 1.0L + drho_ref;

    T drho, u[L::D];
    lbm::moments_shifted<L, T>(f, drho, u);

    const long double tol_drho = K_sum * (eps_t + eps_ld) * S;
    EXPECT_LE(std::abs(static_cast<long double>(drho) - drho_ref), tol_drho)
        << what << ", drho = " << s.drho << ", u = (" << s.ux << ", " << s.uy << ")";
    for (int d = 0; d < L::D; ++d) {
        const long double absu = std::abs(u_ref[d]);
        const long double tol = tol_u_from(K_sum * eps_t * S, eps_t, rho, absu) +
                                tol_u_from(K_sum * eps_ld * S, eps_ld, rho, absu);
        const long double err = std::abs(static_cast<long double>(u[d]) - u_ref[d]);
        EXPECT_LE(err, tol) << what << ", d = " << d << ", drho = " << s.drho << ", u = ("
                            << s.ux << ", " << s.uy << "), err = " << err << ", tol = " << tol;
    }
}

// b. Arithmetic accuracy.
TYPED_TEST(MomentsShifted, ArithmeticAccuracyOnEquilibrium) {
    using T = TypeParam;
    for (const State& s : low_mach_states()) {
        T drho, u[2], f[L::Q];
        inputs_from_state<T>(s, drho, u, f);
        check_arithmetic<T>(f, "equilibrium", s);
    }
}

// Deterministic non-equilibrium perturbation p_q = 1e-3 * w_q * m_q with the
// integer pattern m_q = ((3q + 1) mod 7) - 3, i.e. -2 1 -3 0 3 -1 2 -2 1 for
// q = 0..8. The pattern is not symmetric under q -> opp(q), so it changes both
// drho and u; this is asserted below for every state.
TYPED_TEST(MomentsShifted, ArithmeticAccuracyOnPerturbedPopulations) {
    using T = TypeParam;
    for (const State& s : low_mach_states()) {
        T drho, u[2], f[L::Q];
        inputs_from_state<T>(s, drho, u, f);
        T fp[L::Q];
        long double dsum = 0.0L, dj[L::D] = {0.0L, 0.0L};
        for (int q = 0; q < L::Q; ++q) {
            const int m = ((3 * q + 1) % 7) - 3;
            const T p = T(1e-3) * L::template w<T>(q) * T(m);
            fp[q] = f[q] + p;
            dsum += static_cast<long double>(p);
            for (int d = 0; d < L::D; ++d) dj[d] += L::c(q, d) * static_cast<long double>(p);
        }
        ASSERT_NE(dsum, 0.0L) << "perturbation must change drho";
        ASSERT_TRUE(dj[0] != 0.0L && dj[1] != 0.0L) << "perturbation must change both u";
        check_arithmetic<T>(fp, "perturbed", s);
    }
}

// c. Round trip against the exact state.
//
// f~_q = feq_ld - w_ld, evaluated in long double, is rounded to T. Its moments
// are exactly (drho_s, rho_s u_s) for the second-order equilibrium. Errors:
//   * rounding of each f~_q to T: <= u |f~_q| per term       (K_rnd = 1/2)
//   * error of feq_ld: <= K_ref eps_ld ref_scale_q per term (test_equilibrium.cpp;
//     the subtraction feq - w is exact by Sterbenz, asserted below)
//   * the T arithmetic of moments_shifted: K_sum eps S, tol_u_from above.
// Input errors add to the terms of both sums, so with
//   E = (K_sum + K_rnd) eps_T S + K_ref eps_ld R,   R = sum_q ref_scale_q,
//   tol_drho = E,   tol_u = E (1 + |u|)/rho + K_u eps_T |u|,
// where S = sum_q |f~_q| of the rounded inputs, and rho and u are the state's.
// (The (1+u) factor of the input rounding relative to the exact |f~_q| is
// second order and covered by the headroom in K_sum.)
TYPED_TEST(MomentsShifted, RoundTripAgainstExactState) {
    using T = TypeParam;
    const long double eps_t = this->eps();
    const long double eps_ld = std::numeric_limits<long double>::epsilon();
    for (const State& s : low_mach_states()) {
        const T drho_s = static_cast<T>(s.drho);
        const T u_s[2] = {static_cast<T>(s.ux), static_cast<T>(s.uy)};
        const long double drho_ld = drho_s;
        const long double u_ld[2] = {u_s[0], u_s[1]};
        const long double rho = 1.0L + drho_ld;

        T f[L::Q];
        long double S = 0.0L, R = 0.0L;
        for (int q = 0; q < L::Q; ++q) {
            const long double w = L::w<long double>(q);
            const long double feq = feq_classical_ld(q, drho_ld, u_ld);
            ASSERT_GE(feq, 0.5L * w) << "Sterbenz condition violated, q = " << q;
            ASSERT_LE(feq, 2.0L * w) << "Sterbenz condition violated, q = " << q;
            f[q] = static_cast<T>(feq - w);
            S += std::abs(static_cast<long double>(f[q]));
            R += ref_scale(q, drho_ld, u_ld);
        }

        T drho, u[L::D];
        lbm::moments_shifted<L, T>(f, drho, u);

        const long double E = (K_sum + K_rnd) * eps_t * S + K_ref * eps_ld * R;
        const long double err_rho = std::abs(static_cast<long double>(drho) - drho_ld);
        EXPECT_LE(err_rho, E) << "drho = " << s.drho << ", u = (" << s.ux << ", " << s.uy
                              << "), err = " << err_rho << ", tol = " << E;
        for (int d = 0; d < L::D; ++d) {
            const long double tol = tol_u_from(E, eps_t, rho, std::abs(u_ld[d]));
            const long double err = std::abs(static_cast<long double>(u[d]) - u_ld[d]);
            EXPECT_LE(err, tol) << "d = " << d << ", drho = " << s.drho << ", u = (" << s.ux
                                << ", " << s.uy << "), err = " << err << ", tol = " << tol;
        }
    }
}
