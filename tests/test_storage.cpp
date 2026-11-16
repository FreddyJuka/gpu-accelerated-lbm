// Tests of the storage conversion trait (lbm/storage.hpp).
//
// Every comparison is exact (EXPECT_EQ): load and store are either the
// identity or a single IEEE conversion whose result is known bitwise.

#include <gtest/gtest.h>

#include <cfenv>
#include <cmath>
#include <limits>
#include <type_traits>

#include "lbm/collision/trt.hpp"
#include "lbm/force.hpp"
#include "lbm/storage.hpp"
#include "support/fields.hpp"
#include "support/fp_environment.hpp"

using namespace lbmtest;

// Supported combinations, and only those.
static_assert(lbm::StorageConversion<float, float>::supported);
static_assert(lbm::StorageConversion<double, double>::supported);
static_assert(lbm::StorageConversion<float, double>::supported);
static_assert(!lbm::StorageConversion<double, float>::supported,
              "double storage with float compute must be rejected");
static_assert(!lbm::StorageConversion<long double, long double>::supported);
static_assert(!lbm::StorageConversion<int, float>::supported);
static_assert(!lbm::StorageConversion<unsigned short, float>::supported,
              "a bare 16-bit integer is not a storage format");

// The compute type is the collision policy's; a policy without compute_type
// computes in the storage type.
static_assert(std::is_same_v<lbm::collision_compute_t<lbm::TRT<float>, float>, float>);
static_assert(std::is_same_v<lbm::collision_compute_t<lbm::TRT<double>, double>, double>);
static_assert(std::is_same_v<lbm::collision_compute_t<lbm::TRT<double>, float>, double>);
static_assert(std::is_same_v<
              lbm::collision_compute_t<lbm::TRT<double, lbm::UniformForce<double, 2>>, float>,
              double>);
static_assert(std::is_same_v<lbm::collision_compute_t<IdentityCollision, float>, float>);
static_assert(std::is_same_v<lbm::collision_compute_t<IdentityCollision, double>, double>);

// Return types.
static_assert(std::is_same_v<decltype(lbm::StorageConversion<float, double>::load(0.0f)), double>);
static_assert(std::is_same_v<decltype(lbm::StorageConversion<float, double>::store(0.0)), float>);

namespace {

template <typename T>
class StorageIdentity : public ::testing::Test {};
using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(StorageIdentity, PrecisionTypes);

} // namespace

// a. Storage == Compute: load and store return their argument, sign of zero
// included.
TYPED_TEST(StorageIdentity, LoadAndStoreReturnTheArgument) {
    using T = TypeParam;
    using C = lbm::StorageConversion<T, T>;
    using lim = std::numeric_limits<T>;
    const T values[] = {T(0),       -T(0),         T(1),      T(-1),         T(1) / T(3),
                        lim::min(), lim::denorm_min(), lim::max(), lim::lowest(), lim::epsilon()};
    for (const T v : values) {
        EXPECT_EQ(C::load(v), v);
        EXPECT_EQ(C::store(v), v);
        EXPECT_EQ(std::signbit(C::load(v)), std::signbit(v));
        EXPECT_EQ(std::signbit(C::store(v)), std::signbit(v));
    }
}

// b. float storage, double compute: load is the exact widening, and store
// inverts it for every float.
TEST(StorageFloatDouble, LoadIsExactAndStoreInvertsIt) {
    using C = lbm::StorageConversion<float, double>;
    using lim = std::numeric_limits<float>;
    const float values[] = {0.0f,       -0.0f,             1.0f,       -1.0f,         1.0f / 3.0f,
                            lim::min(), lim::denorm_min(), lim::max(), lim::lowest(), lim::epsilon(),
                            1.0f + lim::epsilon(), 1e-3f, -4.2e-5f};
    for (const float s : values) {
        const double c = C::load(s);
        EXPECT_EQ(c, static_cast<double>(s));
        EXPECT_EQ(std::signbit(c), std::signbit(s));
        EXPECT_EQ(C::store(c), s);
        EXPECT_EQ(std::signbit(C::store(c)), std::signbit(s));
    }
}

// c. store rounds to nearest, ties to even. With e = 2^-23 (the float epsilon,
// spacing of floats in [1, 2)), every input below is exact in double:
//   1 + e/2            tie between 1 (even) and 1 + e (odd)        -> 1
//   1 + 3e/2           tie between 1 + e (odd) and 1 + 2e (even)   -> 1 + 2e
//   1 + e/2 + 2^-52    just above the tie                          -> 1 + e
//   1 + e/2 - 2^-53    just below the tie (exact: 1 + e/2 - 2^-53 has its
//                      lowest bit at 2^-53, within 53 bits below 2)  -> 1
// The rounding mode is asserted.
TEST(StorageFloatDouble, StoreRoundsToNearestTiesToEven) {
    using C = lbm::StorageConversion<float, double>;
    ASSERT_EQ(std::fegetround(), FE_TONEAREST);
    const double e = static_cast<double>(std::numeric_limits<float>::epsilon());
    ASSERT_EQ(e, std::ldexp(1.0, -23));
    const float one = 1.0f;
    const float e_f = std::numeric_limits<float>::epsilon();

    EXPECT_EQ(C::store(1.0 + e / 2), one);
    EXPECT_EQ(C::store(1.0 + 3 * e / 2), one + 2 * e_f);
    EXPECT_EQ(C::store(1.0 + e / 2 + std::ldexp(1.0, -52)), one + e_f);
    EXPECT_EQ(C::store(1.0 + e / 2 - std::ldexp(1.0, -53)), one);
    EXPECT_EQ(C::store(-(1.0 + e / 2)), -one);
    EXPECT_EQ(C::store(-(1.0 + 3 * e / 2)), -(one + 2 * e_f));
}
