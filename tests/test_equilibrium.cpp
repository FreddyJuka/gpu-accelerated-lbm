#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "lbm/equilibrium.hpp"
#include "lbm/lattice/d2q9.hpp"

using L = lbm::D2Q9;

namespace {

constexpr int delta(int a, int b) { return a == b ? 1 : 0; }

struct State {
    double drho;
    double ux, uy;
};

std::vector<State> low_mach_states() {
    const double drhos[] = {0.0, 0.01, -0.02};
    const double us[][2] = {{0.0, 0.0}, {0.05, 0.0}, {0.0, -0.03}, {0.04, 0.07}};
    std::vector<State> s;
    for (double r : drhos)
        for (const auto& u : us) s.push_back({r, u[0], u[1]});
    return s;
}

// The reference below must be more precise than double, or it cannot resolve
// the rounding error of the double-precision path.
static_assert(std::numeric_limits<long double>::digits >= 64,
              "long double must be extended precision for the reference");

// Classical unshifted equilibrium in long double.
long double feq_classical_ld(int q, long double drho, const long double (&u)[2]) {
    const long double rho = 1.0L + drho;
    const long double cs2 = L::cs2<long double>();
    const long double cu = L::c(q, 0) * u[0] + L::c(q, 1) * u[1];
    const long double uu = u[0] * u[0] + u[1] * u[1];
    return L::w<long double>(q) * rho *
           (1.0L + cu / cs2 + cu * cu / (2.0L * cs2 * cs2) - uu / (2.0L * cs2));
}

// Error bound for one computed f~eq_q, with u = eps/2 the unit roundoff and
// every rounding counted as a relative error <= u of the quantity it produces:
//   cu   : c*u products are exact (c in {-1,0,1}); one addition      -> 1
//   t1 = inv_cs2*cu                : cu (1) + product (1)             -> 2
//   t2 = 0.5*inv_cs2*inv_cs2*cu*cu : constants exact; cu (1) + 2 products -> 3
//   t3 = 0.5*inv_cs2*uu            : uu = 2 products + 1 add (2, relative
//                                    since all terms are >= 0) + 1 product -> 3
//   poly = t1 + t2 - t3: two additions; t1 and t2 pass through both,
//   t3 only through the second:  t1: 2+2 = 4, t2: 3+2 = 5, t3: 3+1 = 4,
//   so |err(poly)| <= 5u * (|t1| + |t2| + |t3|).
//   rho = 1 + drho (1) and rho*poly (1) add 2 more: 7u on rho*sum|t|.
//   drho + rho*poly (1 addition) and w*(...) (1 product), plus the rounding
//   of w<T> itself (1): 3 more, giving 10u on the sum of magnitudes
//   (the |drho| term only gets 3u, which 10u covers).
// Hence |err| <= 10u * scale_e = 5 eps * scale_e. K_e = 6 adds one eps of
// headroom for the O(n u) second-order terms of (1+u)^n - 1; it is an
// upper bound, not a fitted value.
constexpr long double K_e = 6.0L;
static_assert(K_e <= 16.0L, "K_e must not exceed 16");

// Sum of the magnitudes of the terms of f~eq_q, evaluated in long double at
// inputs already rounded to the compute type.
long double scale_e(int q, long double drho, const long double (&u)[2]) {
    const long double rho = 1.0L + drho;
    const long double inv_cs2 = L::inv_cs2<long double>();
    const long double cu = L::c(q, 0) * u[0] + L::c(q, 1) * u[1];
    const long double uu = u[0] * u[0] + u[1] * u[1];
    return L::w<long double>(q) *
           (std::abs(drho) + rho * (inv_cs2 * std::abs(cu) + 0.5L * inv_cs2 * inv_cs2 * cu * cu +
                                    0.5L * inv_cs2 * uu));
}

long double sum_scale_e(long double drho, const long double (&u)[2]) {
    long double sum = 0.0L;
    for (int q = 0; q < L::Q; ++q) sum += scale_e(q, drho, u);
    return sum;
}

template <typename T>
T feq_shifted(int q, T drho, const T (&u)[2]) {
    return lbm::equilibrium_shifted<L, T>(q, drho, u);
}

} // namespace

template <typename T>
class EquilibriumShifted : public ::testing::Test {
protected:
    static T eps() { return std::numeric_limits<T>::epsilon(); }

    // Cast a state to compute precision and evaluate all populations.
    static void eval(const State& s, T& drho, T (&u)[2], T (&f)[L::Q]) {
        drho = static_cast<T>(s.drho);
        u[0] = static_cast<T>(s.ux);
        u[1] = static_cast<T>(s.uy);
        for (int q = 0; q < L::Q; ++q) f[q] = lbm::equilibrium_shifted<L, T>(q, drho, u);
    }
};

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
    const long double eps = std::numeric_limits<T>::epsilon();
    const long double tol = (K_e + L::Q) * eps * sum_scale + K_x * eps * scale_expected;
    const long double err = std::abs(sum - expected);
    EXPECT_LE(err, tol) << "drho = " << s.drho << ", u = (" << s.ux << ", " << s.uy
                        << "), err = " << err << ", tol = " << tol;
}

TYPED_TEST(EquilibriumShifted, ZerothMoment) {
    using T = TypeParam;
    for (const State& s : low_mach_states()) {
        T drho, u[2], f[L::Q];
        this->eval(s, drho, u, f);
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
        this->eval(s, drho, u, f);
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
        this->eval(s, drho, u, f);
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
    // error, relative to |feq_classical_ld| (~ w_q), not to scale_e_q. With
    // u_ld = eps_ld/2 and every rounding a relative error <= u_ld:
    //   rho = 1 + drho                       -> 1
    //   cs2<long double> = 1/3 (rounded)     -> 1 wherever cs2 appears
    //   cu : one addition (c*u exact)        -> 1;   uu : 2 products + 1 add -> 2
    //   t1 = cu/cs2             : cu 1 + cs2 1 + division 1               -> 3
    //   t2 = cu*cu/(2 cs2 cs2)  : numerator 1+1+1, denominator 1+1+1
    //                             (2*cs2 is exact), division 1             -> 7
    //   t3 = uu/(2 cs2)         : uu 2 + cs2 1 + division 1                -> 4
    //   S = ((1 + t1) + t2) - t3: three additions; 1 and t1 pass through all
    //   three, t2 through two, t3 through one:
    //   1: 3, t1: 3+3 = 6, t2: 7+2 = 9, t3: 4+1 = 5 -> |err(S)| <= 9u_ld * sum|terms|.
    //   (w*rho)*S : w<long double> 1, rho 1, product w*rho 1, final product 1
    //   -> 9 + 4 = 13u_ld = 6.5 eps_ld on ref_scale.
    // The subtraction feq - w_q adds no error: it is exact by Sterbenz's lemma
    // when w_q/2 <= feq <= 2 w_q, which is asserted for every case below.
    // K_ref = 7 adds headroom for the second-order terms of (1+u)^n - 1;
    // it is an upper bound, not a fitted value.
    constexpr long double K_ref = 7.0L;
    static_assert(K_ref <= 16.0L, "K_ref must not exceed 16");
    const long double eps_ld = std::numeric_limits<long double>::epsilon();
    for (const State& s : low_mach_states()) {
        T drho, u[2], f[L::Q];
        this->eval(s, drho, u, f);
        // Reference evaluated at the inputs already rounded to T.
        const long double drho_ld = static_cast<long double>(drho);
        const long double ud[2] = {static_cast<long double>(u[0]), static_cast<long double>(u[1])};
        const long double rho = 1.0L + drho_ld;
        const long double cs2 = L::cs2<long double>();
        const long double uu = ud[0] * ud[0] + ud[1] * ud[1];
        for (int q = 0; q < L::Q; ++q) {
            const long double w = L::w<long double>(q);
            const long double feq = feq_classical_ld(q, drho_ld, ud);
            ASSERT_GE(feq, 0.5L * w) << "Sterbenz condition violated, q = " << q;
            ASSERT_LE(feq, 2.0L * w) << "Sterbenz condition violated, q = " << q;
            const long double ref = feq - w;
            const long double cu = L::c(q, 0) * ud[0] + L::c(q, 1) * ud[1];
            const long double ref_scale =
                w * rho * (1.0L + std::abs(cu) / cs2 + cu * cu / (2.0L * cs2 * cs2) +
                           uu / (2.0L * cs2));
            const long double tol =
                K_e * static_cast<long double>(this->eps()) * scale_e(q, drho_ld, ud) +
                K_ref * eps_ld * ref_scale;
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
