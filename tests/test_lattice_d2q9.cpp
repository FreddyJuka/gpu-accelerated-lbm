#include <gtest/gtest.h>

#include <limits>

#include "lbm/lattice/d2q9.hpp"

using L = lbm::D2Q9;

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

template <typename T>
constexpr T tol() { return T(16) * std::numeric_limits<T>::epsilon(); }

constexpr int delta(int a, int b) { return a == b ? 1 : 0; }

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
    EXPECT_NEAR(L::template inv_cs2<T>() * L::template cs2<T>(), T(1),
                std::numeric_limits<T>::epsilon());
}

TYPED_TEST(D2Q9Moments, ZerothMomentIsOne) {
    using T = TypeParam;
    T sum = T(0);
    for (int q = 0; q < L::Q; ++q) sum += L::template w<T>(q);
    EXPECT_NEAR(sum, T(1), tol<T>());
}

TYPED_TEST(D2Q9Moments, FirstMomentVanishes) {
    using T = TypeParam;
    for (int a = 0; a < L::D; ++a) {
        T sum = T(0);
        for (int q = 0; q < L::Q; ++q) sum += L::template w<T>(q) * T(L::c(q, a));
        EXPECT_NEAR(sum, T(0), tol<T>()) << "a = " << a;
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
            EXPECT_NEAR(sum, cs2 * T(delta(a, b)), tol<T>())
                << "a = " << a << ", b = " << b;
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
                EXPECT_NEAR(sum, T(0), tol<T>())
                    << "a = " << a << ", b = " << b << ", g = " << g;
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
                    const T expected = cs4 * T(delta(a, b) * delta(g, d) +
                                               delta(a, g) * delta(b, d) +
                                               delta(a, d) * delta(b, g));
                    EXPECT_NEAR(sum, expected, tol<T>())
                        << "a = " << a << ", b = " << b << ", g = " << g << ", d = " << d;
                }
}

TYPED_TEST(D2Q9Moments, WeightsAreSymmetric) {
    using T = TypeParam;
    for (int q = 0; q < L::Q; ++q)
        EXPECT_EQ(L::template w<T>(L::opp(q)), L::template w<T>(q)) << "q = " << q;
}
