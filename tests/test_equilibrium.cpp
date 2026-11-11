#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "lbm/equilibrium.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "support/error_bounds.hpp"
#include "support/fields.hpp"
#include "support/fp_environment.hpp"

using namespace lbmtest;

namespace {

ld sum_scale_e(ld drho, const ld (&u)[2]) {
    ld sum = 0.0L;
    for (int q = 0; q < L::Q; ++q) sum += scale_e(q, drho, u);
    return sum;
}

template <typename T>
T feq_shifted(int q, T drho, const T (&u)[2]) {
    return lbm::equilibrium_shifted<L, T>(q, drho, u);
}

} // namespace

template <typename T>
class EquilibriumShifted : public ::testing::Test {};

using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(EquilibriumShifted, PrecisionTypes);

TYPED_TEST(EquilibriumShifted, RestStateIsExactlyZero) {
    using T = TypeParam;
    const T u[2] = {T(0), T(0)};
    for (int q = 0; q < L::Q; ++q)
        EXPECT_EQ(feq_shifted<T>(q, T(0), u), T(0)) << "q = " << q;
}

// Moment tolerance, shared by the three moment tests:
//
//   tol = (K_e + Q) * eps * sum_q scale_e_q  +  K_x * eps * scale_expected
//
// First term: each computed f~eq_q has error <= K_e*eps*scale_e_q. The factors
// c_q (and c_qa*c_qb) are in {-1,0,1}, so they are exact and do not enlarge
// this. Recursive summation of Q terms adds <= (Q-1)*(eps/2)*sum|f~eq_q|
// <= (Q/2)*eps*sum scale_e_q, because |f~eq_q| <= scale_e_q. K_e + Q is
// therefore an upper bound on both contributions.
//
// Second term: the expected value is itself evaluated in T. scale_expected is
// the sum of the magnitudes of its terms and K_x follows from its operation
// count (u = eps/2; K_x includes headroom for second-order terms of
// (1+u)^n - 1, as for K_e):
//   zeroth : expected = drho, an input, no rounding              -> K_x = 0
//   first  : rho*u_a with rho = 1 + drho (1) and one product (1):
//            2u = 1 eps                                           -> K_x = 2
//   second : drho*cs2*delta + rho*u_a*u_b. First term: cs2<T> (1), two
//            products (the delta factor is exact): 3u. Second term: rho (1),
//            two products: 3u. The final addition (1) applies to both:
//            4u = 2 eps on the sum of magnitudes                  -> K_x = 3
template <typename T>
void expect_moment(long double sum, long double expected, long double sum_scale,
                   long double K_x, long double scale_expected, const State& s) {
    const long double tol = (K_e + L::Q) * kEps<T> * sum_scale + K_x * kEps<T> * scale_expected;
    const long double err = std::abs(sum - expected);
    EXPECT_LE(err, tol) << "drho = " << s.drho << ", u = (" << s.ux << ", " << s.uy
                        << "), err = " << err << ", tol = " << tol;
}

TYPED_TEST(EquilibriumShifted, ZerothMoment) {
    using T = TypeParam;
    for (const State& s : low_mach_states()) {
        T drho, u[2], f[L::Q];
        inputs_from_state<T>(s, drho, u, f);
        const long double drho_ld = drho;
        const long double u_ld[2] = {u[0], u[1]};
        T sum = T(0);
        for (int q = 0; q < L::Q; ++q) sum += f[q];
        expect_moment<T>(sum, drho_ld, sum_scale_e(drho_ld, u_ld), 0.0L, std::abs(drho_ld), s);
    }
}

TYPED_TEST(EquilibriumShifted, FirstMoment) {
    using T = TypeParam;
    for (const State& s : low_mach_states()) {
        T drho, u[2], f[L::Q];
        inputs_from_state<T>(s, drho, u, f);
        const long double drho_ld = drho;
        const long double u_ld[2] = {u[0], u[1]};
        const T rho = T(1) + drho;
        for (int a = 0; a < L::D; ++a) {
            T sum = T(0);
            for (int q = 0; q < L::Q; ++q) sum += f[q] * T(L::c(q, a));
            const T expected = rho * u[a];
            expect_moment<T>(sum, expected, sum_scale_e(drho_ld, u_ld), 2.0L,
                             (1.0L + std::abs(drho_ld)) * std::abs(u_ld[a]), s);
        }
    }
}

TYPED_TEST(EquilibriumShifted, SecondMoment) {
    using T = TypeParam;
    const T cs2 = L::template cs2<T>();
    for (const State& s : low_mach_states()) {
        T drho, u[2], f[L::Q];
        inputs_from_state<T>(s, drho, u, f);
        const long double drho_ld = drho;
        const long double u_ld[2] = {u[0], u[1]};
        const T rho = T(1) + drho;
        for (int a = 0; a < L::D; ++a)
            for (int b = 0; b < L::D; ++b) {
                T sum = T(0);
                for (int q = 0; q < L::Q; ++q) sum += f[q] * T(L::c(q, a) * L::c(q, b));
                const T expected = drho * cs2 * T(delta(a, b)) + rho * u[a] * u[b];
                const long double scale_expected =
                    std::abs(drho_ld) * L::cs2<long double>() * delta(a, b) +
                    (1.0L + std::abs(drho_ld)) * std::abs(u_ld[a]) * std::abs(u_ld[b]);
                expect_moment<T>(sum, expected, sum_scale_e(drho_ld, u_ld), 3.0L, scale_expected,
                                 s);
            }
    }
}

TYPED_TEST(EquilibriumShifted, AgreesWithShiftedReferenceInLongDouble) {
    using T = TypeParam;
    // The reference f~eq_ref = feq_classical_ld - w_ld carries its own rounding
    // error, relative to ref_scale (~ w_q), not to scale_e_q: K_feq_ld eps_ld
    // ref_scale (derivation at K_feq_ld, error_bounds.hpp). It needs
    // rho >= kRhoMin, asserted for every state, and the subtraction feq - w_q
    // is exact by Sterbenz's lemma when w_q/2 <= feq <= 2 w_q, asserted for
    // every case below.
    for (const State& s : low_mach_states()) {
        T drho, u[2], f[L::Q];
        inputs_from_state<T>(s, drho, u, f);
        // Reference evaluated at the inputs already rounded to T.
        const long double drho_ld = static_cast<long double>(drho);
        const long double ud[2] = {static_cast<long double>(u[0]), static_cast<long double>(u[1])};
        ASSERT_GE(1.0L + drho_ld, kRhoMin) << "rho below kRhoMin";
        for (int q = 0; q < L::Q; ++q) {
            const long double w = L::w<long double>(q);
            const long double feq = feq_classical_ld(q, drho_ld, ud);
            ASSERT_GE(feq, 0.5L * w) << "Sterbenz condition violated, q = " << q;
            ASSERT_LE(feq, 2.0L * w) << "Sterbenz condition violated, q = " << q;
            const long double ref = feq - w;
            const long double tol =
                K_e * kEps<T> * scale_e(q, drho_ld, ud) + K_feq_ld * kEpsLd * ref_scale(q, drho_ld, ud);
            const long double err = std::abs(static_cast<long double>(f[q]) - ref);
            EXPECT_LE(err, tol) << "q = " << q << ", drho = " << s.drho << ", u = (" << s.ux
                                << ", " << s.uy << "), err = " << err << ", tol = " << tol;
        }
    }
}

TYPED_TEST(EquilibriumShifted, VelocityReversalSymmetry) {
    using T = TypeParam;
    for (const State& s : low_mach_states()) {
        const T drho = static_cast<T>(s.drho);
        const T u[2] = {static_cast<T>(s.ux), static_cast<T>(s.uy)};
        const T mu[2] = {-u[0], -u[1]};
        for (int q = 0; q < L::Q; ++q)
            EXPECT_EQ(feq_shifted<T>(L::opp(q), drho, mu), feq_shifted<T>(q, drho, u))
                << "q = " << q << ", drho = " << s.drho << ", u = (" << s.ux << ", " << s.uy
                << ")";
    }
}
