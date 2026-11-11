#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "lbm/equilibrium.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "lbm/moments.hpp"
#include "support/error_bounds.hpp"
#include "support/fields.hpp"
#include "support/fp_environment.hpp"

using namespace lbmtest;

namespace {

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
// Error bounds. u = eps/2 is the unit roundoff of the type in question. The
// constants K_sum (summation of Q terms), K_u (division and product
// u_d = j_d * inv_rho) and K_feq_ld (long double reference of the equilibrium)
// and tol_u_from are defined with their derivations in support/error_bounds.hpp.
// ---------------------------------------------------------------------------
// Rounding of each input f~_q to T, at most u |f~_q| = 0.5 eps |f~_q|, enters
// as an additional absolute error of the terms of both sums: K_rnd = 1/2.
constexpr ld K_rnd = 0.5L;
static_assert(K_rnd <= 16.0L, "every factor must not exceed 16");

} // namespace

template <typename T>
class MomentsShifted : public ::testing::Test {};

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
    const long double eps_t = kEps<T>;
    const long double eps_ld = kEpsLd;

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

// Deterministic non-equilibrium perturbation (perturbation<T>, fields.hpp). The
// pattern is not symmetric under q -> opp(q), so it changes both drho and u;
// this is asserted below for every state.
TYPED_TEST(MomentsShifted, ArithmeticAccuracyOnPerturbedPopulations) {
    using T = TypeParam;
    for (const State& s : low_mach_states()) {
        T drho, u[2], f[L::Q];
        inputs_from_state<T>(s, drho, u, f);
        T fp[L::Q];
        long double dsum = 0.0L, dj[L::D] = {0.0L, 0.0L};
        for (int q = 0; q < L::Q; ++q) {
            const T p = perturbation<T>(q);
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
//   * error of feq_ld: <= K_feq_ld eps_ld ref_scale_q per term (the subtraction
//     feq - w is exact by Sterbenz, asserted below; rho >= kRhoMin, asserted)
//   * the T arithmetic of moments_shifted: K_sum eps S, tol_u_from above.
// Input errors add to the terms of both sums, so with
//   E = (K_sum + K_rnd) eps_T S + K_feq_ld eps_ld R,   R = sum_q ref_scale_q,
//   tol_drho = E,   tol_u = E (1 + |u|)/rho + K_u eps_T |u|,
// where S = sum_q |f~_q| of the rounded inputs, and rho and u are the state's.
// (The (1+u) factor of the input rounding relative to the exact |f~_q| is
// second order and covered by the headroom in K_sum.)
TYPED_TEST(MomentsShifted, RoundTripAgainstExactState) {
    using T = TypeParam;
    const long double eps_t = kEps<T>;
    const long double eps_ld = kEpsLd;
    for (const State& s : low_mach_states()) {
        const T drho_s = static_cast<T>(s.drho);
        const T u_s[2] = {static_cast<T>(s.ux), static_cast<T>(s.uy)};
        const long double drho_ld = drho_s;
        const long double u_ld[2] = {u_s[0], u_s[1]};
        const long double rho = 1.0L + drho_ld;
        ASSERT_GE(rho, kRhoMin) << "rho below kRhoMin";

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

        const long double E = (K_sum + K_rnd) * eps_t * S + K_feq_ld * eps_ld * R;
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
