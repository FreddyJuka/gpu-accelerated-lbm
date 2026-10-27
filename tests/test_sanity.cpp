#include <gtest/gtest.h>

// Toolchain check: GoogleTest builds, links, and runs.
TEST(Sanity, BasicArithmetic) {
    EXPECT_EQ(2 + 2, 4);
}

// Typed-test check: the same test runs for each precision type.
template <typename T>
class SanityTyped : public ::testing::Test {};

using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SanityTyped, PrecisionTypes);

TYPED_TEST(SanityTyped, OneThirdTimesThree) {
    const TypeParam third = TypeParam(1) / TypeParam(3);
    EXPECT_NEAR(third * TypeParam(3), TypeParam(1), TypeParam(1e-6));
}
