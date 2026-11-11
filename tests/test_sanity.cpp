#include <gtest/gtest.h>

#include "support/fp_environment.hpp"

// Toolchain check: GoogleTest builds, links, and runs.
TEST(Sanity, BasicArithmetic) {
    EXPECT_EQ(2 + 2, 4);
}

// Typed-test check: the same test runs for each precision type.
template <typename T>
class SanityTyped : public ::testing::Test {};

using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SanityTyped, PrecisionTypes);

// Exact in float and double: fl(1/3) = 11184811 * 2^-25 (float) and
// 6004799503160661 * 2^-54 (double); three times these is 1 + 2^-25 resp.
// 1 - 2^-54, which rounds (to nearest, ties to even) to 1 in both types. This
// relies on the IEEE environment checked in support/fp_environment.hpp.
TYPED_TEST(SanityTyped, OneThirdTimesThree) {
    const TypeParam third = TypeParam(1) / TypeParam(3);
    EXPECT_EQ(third * TypeParam(3), TypeParam(1));
}
