#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "lbm/lattice/d2q9.hpp"
#include "support/error_bounds.hpp"
#include "support/fields.hpp"
#include "support/fp_environment.hpp"

using namespace lbmtest;

namespace {

// Compile-time check: opposite table is an involution and reverses velocities.
constexpr bool opposites_consistent() {
    for (int q = 0; q < L::Q; ++q) {
        if (L::opp(L::opp(q)) != q) return false;
        for (int d = 0; d < L::D; ++d) {
            if (L::c(L::opp(q), d) != -L::c(q, d)) return false;
        }
    }
    return true;
}
static_assert(opposites_consistent(), "D2Q9 opposite table is inconsistent");

// ---------------------------------------------------------------------------
// Tolerance of the lattice moment tests, from operation counts (u = eps/2).
//
// Each test forms sum_q w_q g_q in T, with g_q = product of 0 to 4 integer
// velocity components, g_q in {-1, 0, 1}. The products g_q (and w_q g_q) are
// exact, so the only roundings are
//   - the representation of the weights, w_T(q) = w_q (1 + d_q), |d_q| <= u
//     (T(4), T(9), T(1), T(36) are exact, then one division): K_w u S_g;
//   - the recursive summation of Q terms: K_acc u S_g (1 + O(Q u));
//   - 1 u S_g of headroom for the O(Q u) second-order factor of the summation
//     and of the weight errors in the term magnitudes. The exact second-order
//     excess is at most (K_acc^2 u / (1 - K_acc u) (1 + u) + K_acc u) u S_g
//     <= 2 (K_acc + 1)^2 u^2 S_g, below the 1 u S_g allowed whenever
//     2 (K_acc + 1)^2 u <= 1 (static_assert below, for both types).
// with S_g = sum_q w_q |g_q|, a sum of magnitudes (it does not shrink when the
// terms cancel). The expected value is the exact moment, except that the
// second and fourth moment tests compare against a value computed in T:
//   second : cs2<T> * T(delta)          cs2<T> = fl(1/3): 1 rounding
//   fourth : cs2<T> * cs2<T> * T(k)     two cs2<T> (2), their product (1), the
//                                       product with T(k), k in {0, 1, 3} (1)
// which adds K_exp u |exact| with K_exp = 1 and 4 respectively.
// ---------------------------------------------------------------------------
constexpr ld K_exp_second = 1.0L;
constexpr ld K_exp_fourth = 4.0L;
constexpr ld K_headroom = 1.0L;  // second-order terms of the summation, in units of u
static_assert(K_acc + K_w + K_headroom <= 16.0L && K_exp_second <= 16.0L && K_exp_fourth <= 16.0L,
              "every factor must not exceed 16");
static_assert(2.0L * (K_acc + 1.0L) * (K_acc + 1.0L) * kU<float> <= 1.0L &&
                  2.0L * (K_acc + 1.0L) * (K_acc + 1.0L) * kU<double> <= 1.0L,
              "the second-order excess of the summation must fit in K_headroom");

// Checks |sum - expected| <= tol where `exact` is the exact moment, S_g the
// sum of the magnitudes of the terms and K_exp the count of roundings in the
// computation of `expected` (0 when it is an exact integer).
template <typename T>
void expect_lattice_moment(T sum, T expected, ld exact, ld K_exp, ld S_g,
                           const std::string& what) {
    const ld tol = (K_acc + K_w + K_headroom) * kU<T> * S_g + K_exp * kU<T> * std::abs(exact);
    const ld err = std::abs(static_cast<ld>(sum) - static_cast<ld>(expected));
    EXPECT_LE(err, tol) << what << ", err = " << err << ", tol = " << tol;
}

// S_g for g_q = prod_i c_{q, idx[i]} over n components.
inline ld magnitude_sum(int n, const int (&idx)[4]) {
    ld S = 0;
    for (int q = 0; q < L::Q; ++q) {
        int g = 1;
        for (int i = 0; i < n; ++i) g *= L::c(q, idx[i]);
        S += L::w<ld>(q) * std::abs(static_cast<ld>(g));
    }
    return S;
}

} // namespace

// ---------------------------------------------------------------------------
// Structural properties (integer, precision-independent)
// ---------------------------------------------------------------------------
TEST(D2Q9Structure, DimensionsAndRestDirection) {
    EXPECT_EQ(L::D, 2);
    EXPECT_EQ(L::Q, 9);
    EXPECT_EQ(L::cx(0), 0);
    EXPECT_EQ(L::cy(0), 0);
    EXPECT_EQ(L::opp(0), 0);
}

TEST(D2Q9Structure, VelocitiesAreUnique) {
    for (int p = 0; p < L::Q; ++p)
        for (int q = p + 1; q < L::Q; ++q)
            EXPECT_FALSE(L::cx(p) == L::cx(q) && L::cy(p) == L::cy(q))
                << "directions " << p << " and " << q << " coincide";
}

TEST(D2Q9Structure, OppositeDirections) {
    for (int q = 0; q < L::Q; ++q) {
        EXPECT_EQ(L::opp(L::opp(q)), q) << "q = " << q;
        EXPECT_EQ(L::cx(L::opp(q)), -L::cx(q)) << "q = " << q;
        EXPECT_EQ(L::cy(L::opp(q)), -L::cy(q)) << "q = " << q;
    }
}

// ---------------------------------------------------------------------------
// Moment isotropy, checked in each precision
// ---------------------------------------------------------------------------
template <typename T>
class D2Q9Moments : public ::testing::Test {};

using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(D2Q9Moments, PrecisionTypes);

TYPED_TEST(D2Q9Moments, InverseSoundSpeedIsExact) {
    using T = TypeParam;
    EXPECT_EQ(L::template inv_cs2<T>(), T(3));
    // cs2 = fl(1/3) = (1/3)(1 + d), |d| <= u, so the exact product 3 cs2 = 1 + d
    // lies within u of 1; the nearest representable numbers are 1 - u (below 1,
    // spacing eps/2) and 1, 1 + eps (above 1, spacing eps), so the rounded
    // product is within u of 1.
    EXPECT_LE(std::abs(static_cast<ld>(L::template inv_cs2<T>() * L::template cs2<T>()) - 1.0L),
              kU<T>);
}

TYPED_TEST(D2Q9Moments, ZerothMomentIsOne) {
    using T = TypeParam;
    T sum = T(0);
    for (int q = 0; q < L::Q; ++q) sum += L::template w<T>(q);
    const int idx[4] = {0, 0, 0, 0};
    expect_lattice_moment<T>(sum, T(1), 1.0L, 0.0L, magnitude_sum(0, idx), "zeroth");
}

TYPED_TEST(D2Q9Moments, FirstMomentVanishes) {
    using T = TypeParam;
    for (int a = 0; a < L::D; ++a) {
        T sum = T(0);
        for (int q = 0; q < L::Q; ++q) sum += L::template w<T>(q) * T(L::c(q, a));
        const int idx[4] = {a, 0, 0, 0};
        expect_lattice_moment<T>(sum, T(0), 0.0L, 0.0L, magnitude_sum(1, idx),
                                 "a = " + std::to_string(a));
    }
}

TYPED_TEST(D2Q9Moments, SecondMomentIsCs2Delta) {
    using T = TypeParam;
    const T cs2 = L::template cs2<T>();
    for (int a = 0; a < L::D; ++a)
        for (int b = 0; b < L::D; ++b) {
            T sum = T(0);
            for (int q = 0; q < L::Q; ++q)
                sum += L::template w<T>(q) * T(L::c(q, a) * L::c(q, b));
            const int idx[4] = {a, b, 0, 0};
            expect_lattice_moment<T>(sum, cs2 * T(delta(a, b)), L::cs2<ld>() * delta(a, b),
                                     K_exp_second, magnitude_sum(2, idx),
                                     "a = " + std::to_string(a) + ", b = " + std::to_string(b));
        }
}

TYPED_TEST(D2Q9Moments, ThirdMomentVanishes) {
    using T = TypeParam;
    for (int a = 0; a < L::D; ++a)
        for (int b = 0; b < L::D; ++b)
            for (int g = 0; g < L::D; ++g) {
                T sum = T(0);
                for (int q = 0; q < L::Q; ++q)
                    sum += L::template w<T>(q) * T(L::c(q, a) * L::c(q, b) * L::c(q, g));
                const int idx[4] = {a, b, g, 0};
                expect_lattice_moment<T>(sum, T(0), 0.0L, 0.0L, magnitude_sum(3, idx),
                                         "a = " + std::to_string(a) + ", b = " + std::to_string(b) +
                                             ", g = " + std::to_string(g));
            }
}

TYPED_TEST(D2Q9Moments, FourthMomentIsIsotropic) {
    using T = TypeParam;
    const T cs4 = L::template cs2<T>() * L::template cs2<T>();
    for (int a = 0; a < L::D; ++a)
        for (int b = 0; b < L::D; ++b)
            for (int g = 0; g < L::D; ++g)
                for (int d = 0; d < L::D; ++d) {
                    T sum = T(0);
                    for (int q = 0; q < L::Q; ++q)
                        sum += L::template w<T>(q) *
                               T(L::c(q, a) * L::c(q, b) * L::c(q, g) * L::c(q, d));
                    const int k = delta(a, b) * delta(g, d) + delta(a, g) * delta(b, d) +
                                  delta(a, d) * delta(b, g);
                    const int idx[4] = {a, b, g, d};
                    expect_lattice_moment<T>(
                        sum, cs4 * T(k), L::cs2<ld>() * L::cs2<ld>() * k, K_exp_fourth,
                        magnitude_sum(4, idx),
                        "a = " + std::to_string(a) + ", b = " + std::to_string(b) +
                            ", g = " + std::to_string(g) + ", d = " + std::to_string(d));
                }
}

TYPED_TEST(D2Q9Moments, WeightsAreSymmetric) {
    using T = TypeParam;
    for (int q = 0; q < L::Q; ++q)
        EXPECT_EQ(L::template w<T>(L::opp(q)), L::template w<T>(q)) << "q = " << q;
}
