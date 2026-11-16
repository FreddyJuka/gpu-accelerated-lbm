// Tests of the storage conversion trait (lbm/storage.hpp).
//
// Every comparison of a conversion result is exact (EXPECT_EQ): load and store
// are the identity, a single IEEE conversion or, for P16, an integer
// computation, and the result is known bitwise in each case. The only
// inequality is the rounding-error bound of the P16 store, which is the
// property the accuracy analysis uses.

#include <gtest/gtest.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "lbm/collision/trt.hpp"
#include "lbm/force.hpp"
#include "lbm/storage.hpp"
#include "support/error_bounds.hpp"
#include "support/fields.hpp"
#include "support/fp_environment.hpp"
#include "support/p16_reference.hpp"

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

static_assert(lbm::StorageConversion<lbm::P16, float>::supported);
static_assert(!lbm::StorageConversion<lbm::P16, double>::supported,
              "P16 storage is defined for float compute only");
static_assert(!lbm::StorageConversion<lbm::P16, lbm::P16>::supported,
              "P16 is not a compute type");
static_assert(std::is_trivially_copyable_v<lbm::P16> && sizeof(lbm::P16) == 2);
static_assert(std::is_same_v<lbm::collision_compute_t<lbm::TRT<float>, lbm::P16>, float>);

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
static_assert(std::is_same_v<decltype(lbm::StorageConversion<lbm::P16, float>::load(lbm::P16{})), float>);
static_assert(std::is_same_v<decltype(lbm::StorageConversion<lbm::P16, float>::store(0.0f)), lbm::P16>);

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

// ---------------------------------------------------------------------------
// P16: 16-bit storage, float compute
// ---------------------------------------------------------------------------
// The references (support/p16_reference.hpp) are written from the definition
// of the format (sign, 4 exponent bits, 11 mantissa bits; see lbm/storage.hpp)
// with std::ldexp on integers and comparisons of exact long double values.
// They share no code with load and store.
namespace {

using C16 = lbm::StorageConversion<lbm::P16, float>;

constexpr std::uint32_t kInvalid = kP16Invalid;

std::uint16_t store_bits(float x) { return C16::store(x).bits; }
float load_bits(std::uint32_t p) { return C16::load(lbm::P16{static_cast<std::uint16_t>(p)}); }

float float_of(std::uint32_t b) {
    float x;
    std::memcpy(&x, &b, sizeof x);
    return x;
}

// Inputs for the rounding tests, both signs of each:
//  - every 1021st binary32 bit pattern (1021 is prime, so the low mantissa
//    bits take all values): 4.2 million values over the whole range, zeros,
//    subnormals, infinities and NaNs included;
//  - for every finite magnitude pattern k: its value, the midpoint to the next
//    value (a tie, exactly representable in binary32: 13 significant bits) and
//    the binary32 neighbours of both. These are the points where the result
//    changes.
template <typename Fn>
void for_each_rounding_input(const std::vector<ld>& values, Fn&& fn) {
    for (std::uint64_t b = 0; b < (std::uint64_t{1} << 32); b += 1021u) fn(float_of(static_cast<std::uint32_t>(b)));
    const float inf = std::numeric_limits<float>::infinity();
    for (std::uint32_t k = 0; k < kInvalid; ++k) {
        const float v = static_cast<float>(values[k]);
        const float mid = static_cast<float>((values[k] + values[k + 1]) / 2.0L);
        for (const float c : {v, mid})
            for (const float x : {std::nextafter(c, -inf), c, std::nextafter(c, inf)}) {
                fn(x);
                fn(-x);
            }
    }
}

} // namespace

// a. load decodes every one of the 65536 patterns to the value the format
// defines, exactly, with the sign bit (zeros included); the two invalid
// patterns load as NaN.
TEST(StorageP16, LoadDecodesEveryPattern) {
    int nans = 0;
    for (std::uint32_t p = 0; p < 0x10000u; ++p) {
        const float got = load_bits(p);
        if ((p & 0x7FFFu) == kInvalid) {
            ASSERT_TRUE(std::isnan(got)) << "pattern " << p;
            ++nans;
            continue;
        }
        const bool negative = (p & 0x8000u) != 0u;
        const ld expected = negative ? -p16_magnitude_value(p & 0x7FFFu) : p16_magnitude_value(p & 0x7FFFu);
        ASSERT_EQ(static_cast<ld>(got), expected) << "pattern " << p;
        ASSERT_EQ(std::signbit(got), negative) << "pattern " << p;
    }
    EXPECT_EQ(nans, 2);
    // The documented range.
    EXPECT_EQ(load_bits(0x0001u), std::ldexp(1.0f, -25));               // smallest subnormal
    EXPECT_EQ(load_bits(0x07FFu), std::ldexp(2047.0f, -25));            // largest subnormal
    EXPECT_EQ(load_bits(0x0800u), std::ldexp(1.0f, -14));               // smallest normal
    EXPECT_EQ(load_bits(0x7800u), 1.0f);
    EXPECT_EQ(load_bits(0x7FFEu), 2.0f - std::ldexp(1.0f, -10));        // largest finite
}

// b. The magnitude increases strictly with the 15-bit pattern, so the patterns
// of one sign are ordered like the values (used by the rounding reference and
// by the monotonicity test).
TEST(StorageP16, ValuesIncreaseWithThePattern) {
    for (std::uint32_t k = 0; k + 1 < kInvalid; ++k)
        ASSERT_LT(load_bits(k), load_bits(k + 1)) << "pattern " << k;
    for (std::uint32_t k = 0; k < kInvalid; ++k) ASSERT_LT(p16_magnitude_value(k), p16_magnitude_value(k + 1));
}

// c. store inverts load for every pattern: bitwise for the 65534 finite ones;
// an invalid pattern comes back as an invalid pattern.
TEST(StorageP16, StoreInvertsLoadForEveryPattern) {
    for (std::uint32_t p = 0; p < 0x10000u; ++p) {
        const std::uint16_t back = store_bits(load_bits(p));
        if ((p & 0x7FFFu) == kInvalid)
            ASSERT_EQ(back & 0x7FFFu, kInvalid) << "pattern " << p;
        else
            ASSERT_EQ(back, p) << "pattern " << p;
    }
}

// d. store rounds to nearest, ties to even: equal to the exact reference for
// 4.2 million binary32 values spread over all bit patterns and for the
// neighbourhood of every representable value and every tie.
TEST(StorageP16, StoreRoundsToNearestEven) {
    const std::vector<ld> values = p16_magnitude_values();
    long checked = 0, ties = 0, mismatches = 0;
    for_each_rounding_input(values, [&](float x) {
        const std::uint16_t expected = p16_reference_store(x, values);
        const std::uint16_t got = store_bits(x);
        ++checked;
        if (got != expected && ++mismatches <= 10)
            ADD_FAILURE() << "x = " << std::hexfloat << x << std::defaultfloat << ": got " << got
                          << ", expected " << expected;
        // Count the exact ties among the finite results (guard: they must occur).
        const ld a = std::abs(static_cast<ld>(x));
        if (std::isfinite(x) && a < values[kInvalid]) {
            const std::uint32_t k = static_cast<std::uint32_t>(
                std::upper_bound(values.begin(), values.end(), a) - values.begin() - 1);
            if (2.0L * a == values[k] + values[k + 1]) ++ties;
        }
    });
    EXPECT_EQ(mismatches, 0);
    EXPECT_GT(checked, 4000000);
    EXPECT_GE(ties, 2 * 32767);  // every midpoint, both signs
}

// e. Monotonicity and symmetry. Over the non-negative binary32 values in
// increasing order (every 509th bit pattern from +0 to +inf, 4.2 million
// values) the magnitude pattern never decreases; the invalid pattern is the
// largest, so this holds through the overflow as well. With test b, the loaded
// value is then a non-decreasing function of x wherever it is finite. A
// negative input gives the same magnitude pattern with the sign bit set.
TEST(StorageP16, StoreIsMonotonicAndOdd) {
    std::uint32_t prev = 0;
    long steps_up = 0;
    for (std::uint32_t b = 0; b <= 0x7F800000u; b += 509u) {
        const float x = float_of(b);
        const std::uint32_t h = store_bits(x);
        ASSERT_EQ(h & 0x8000u, 0u) << std::hexfloat << x;
        ASSERT_GE(h, prev) << std::hexfloat << x;
        if (h > prev) ++steps_up;
        prev = h;
        ASSERT_EQ(store_bits(-x), h | 0x8000u) << std::hexfloat << x;
    }
    EXPECT_EQ(prev, kInvalid);
    // The sweep is fine enough to produce every magnitude pattern. In
    // [2^-26, 2) one P16 step spans at least 2^12 binary32 values (the 12 low
    // significand bits) and the stride is 509 < 2^12; the subnormal steps are
    // wider still.
    EXPECT_EQ(steps_up, static_cast<long>(kInvalid));
}

// f. Special values: zeros, underflow, subnormals, the ends of the range,
// infinities and NaN, as documented in lbm/storage.hpp.
TEST(StorageP16, SpecialValues) {
    using lim = std::numeric_limits<float>;
    const float floor = std::ldexp(1.0f, -26);       // half the subnormal spacing
    const float step = std::ldexp(1.0f, -25);        // subnormal spacing
    const float min_normal = std::ldexp(1.0f, -14);
    const float max_finite = 2.0f - std::ldexp(1.0f, -10);
    const float max_rounding = max_finite + std::ldexp(1.0f, -12);  // tie with 2 - 2^-11
    ASSERT_EQ(static_cast<ld>(max_rounding), kP16MaxRounding);

    // Zeros keep their sign, in both directions.
    EXPECT_EQ(store_bits(0.0f), 0x0000u);
    EXPECT_EQ(store_bits(-0.0f), 0x8000u);
    EXPECT_FALSE(std::signbit(load_bits(0x0000u)));
    EXPECT_TRUE(std::signbit(load_bits(0x8000u)));
    EXPECT_EQ(load_bits(0x8000u), 0.0f);

    // Underflow to a zero of the same sign: binary32 subnormals, the smallest
    // normal float, everything up to and including the tie 2^-26 (even: zero).
    for (const float x : {lim::denorm_min(), lim::min(), 1e-30f, std::nextafter(floor, 0.0f), floor}) {
        EXPECT_EQ(store_bits(x), 0x0000u) << x;
        EXPECT_EQ(store_bits(-x), 0x8000u) << x;
    }
    // Just above the tie: the smallest subnormal.
    EXPECT_EQ(store_bits(std::nextafter(floor, 1.0f)), 0x0001u);
    EXPECT_EQ(store_bits(step), 0x0001u);
    // 1.5 steps is a tie between 1 (odd) and 2 (even); 2.5 steps between 2 and 3.
    EXPECT_EQ(store_bits(1.5f * step), 0x0002u);
    EXPECT_EQ(store_bits(2.5f * step), 0x0002u);
    EXPECT_EQ(store_bits(-2.5f * step), 0x8002u);

    // Subnormal / normal boundary: the largest subnormal rounds up into the
    // smallest normal number through the carry.
    EXPECT_EQ(store_bits(min_normal), 0x0800u);
    EXPECT_EQ(store_bits(std::nextafter(min_normal, 0.0f)), 0x0800u);
    EXPECT_EQ(store_bits(min_normal - step), 0x07FFu);
    EXPECT_EQ(store_bits(min_normal - 0.5f * step), 0x0800u);  // tie 2047.5 -> 2048 (even)

    // Normal range: a carry out of the mantissa moves to the next binade.
    EXPECT_EQ(store_bits(1.0f), 0x7800u);
    EXPECT_EQ(store_bits(1.0f - std::ldexp(1.0f, -13)), 0x7800u);  // tie 4095.5 -> 4096 (even)
    EXPECT_EQ(store_bits(1.0f + std::ldexp(1.0f, -11)), 0x7801u);
    EXPECT_EQ(store_bits(1.0f + std::ldexp(1.0f, -12)), 0x7800u);  // tie -> even
    EXPECT_EQ(store_bits(1.0f + 3.0f * std::ldexp(1.0f, -12)), 0x7802u);  // tie -> even

    // Upper end. The largest finite value, the tie above it (rounds down to
    // the even pattern), and the first value that overflows.
    EXPECT_EQ(store_bits(max_finite), 0x7FFEu);
    EXPECT_EQ(store_bits(max_rounding), 0x7FFEu);
    EXPECT_EQ(store_bits(-max_rounding), 0xFFFEu);
    EXPECT_EQ(store_bits(std::nextafter(max_rounding, 4.0f)), 0x7FFFu);
    EXPECT_EQ(store_bits(2.0f), 0x7FFFu);
    EXPECT_EQ(store_bits(-2.0f), 0xFFFFu);
    EXPECT_EQ(store_bits(lim::max()), 0x7FFFu);
    EXPECT_EQ(store_bits(lim::lowest()), 0xFFFFu);

    // Infinities and NaN are stored as the invalid pattern, which loads as NaN.
    EXPECT_EQ(store_bits(lim::infinity()), 0x7FFFu);
    EXPECT_EQ(store_bits(-lim::infinity()), 0xFFFFu);
    EXPECT_EQ(store_bits(lim::quiet_NaN()) & 0x7FFFu, kInvalid);
    EXPECT_TRUE(std::isnan(load_bits(0x7FFFu)));
    EXPECT_TRUE(std::isnan(load_bits(0xFFFFu)));
    EXPECT_TRUE(std::isnan(C16::load(C16::store(2.0f))));
    EXPECT_TRUE(std::isnan(C16::load(C16::store(lim::infinity()))));

    // An all-zero buffer is the rest state.
    lbm::P16 zeroed;
    std::memset(&zeroed, 0, sizeof zeroed);
    EXPECT_EQ(C16::load(zeroed), 0.0f);
    EXPECT_FALSE(std::signbit(C16::load(zeroed)));
}

// g. Rounding-error bound of one store (the bound the accuracy analysis
// uses), for the inputs of test d. Derivation (support/error_bounds.hpp): the
// result is the nearest format value, so the error is at most half the local
// spacing:
//   kP16MinNormal <= |x| <= kP16MaxRounding:  |load(store(x)) - x| <= kUP16 |x|
//   |x| < kP16MinNormal:                      |load(store(x)) - x| <= kP16Floor.
// The error is evaluated exactly in long double. y = load(store(x)) and x are
// exact long double values; if y == 0 the error is |x|; otherwise y and x have
// the same sign and y/2 <= x <= 2y in magnitude, so the difference is exact by
// Sterbenz's lemma. Both preconditions are asserted. kUP16 |x| is a
// power-of-two scaling, exact as well, so the comparison involves no rounding.
// Beyond kP16MaxRounding the result must be the invalid pattern.
TEST(StorageP16, StoreErrorBound) {
    const std::vector<ld> values = p16_magnitude_values();
    long normal = 0, subnormal = 0, overflow = 0, violations = 0;
    ld worst_rel = 0, worst_abs = 0;
    for_each_rounding_input(values, [&](float x) {
        if (!std::isfinite(x)) return;
        const float y = C16::load(C16::store(x));
        const ld a = std::abs(static_cast<ld>(x));
        if (a > kP16MaxRounding) {
            ++overflow;
            if (!std::isnan(y) && ++violations <= 10) ADD_FAILURE() << "no overflow for " << std::hexfloat << x;
            return;
        }
        if (!std::isfinite(y)) {
            if (++violations <= 10) ADD_FAILURE() << "non-finite result for " << std::hexfloat << x;
            return;
        }
        ld err;
        if (y == 0.0f) {
            err = a;
        } else {
            const ld b = std::abs(static_cast<ld>(y));
            // Sterbenz preconditions.
            if (std::signbit(y) != std::signbit(x) || a < b / 2.0L || a > 2.0L * b) {
                if (++violations <= 10) ADD_FAILURE() << "Sterbenz precondition fails for " << std::hexfloat << x;
                return;
            }
            err = std::abs(static_cast<ld>(y) - static_cast<ld>(x));
        }
        bool ok;
        if (a >= kP16MinNormal) {
            ++normal;
            ok = err <= K_store * kUP16 * a;
            worst_rel = std::max(worst_rel, err / a);
        } else {
            ++subnormal;
            ok = err <= K_store * kP16Floor;
            worst_abs = std::max(worst_abs, err);
        }
        if (!ok && ++violations <= 10) ADD_FAILURE() << "bound violated for " << std::hexfloat << x;
    });
    EXPECT_EQ(violations, 0);
    EXPECT_GT(normal, 100000);
    EXPECT_GT(subnormal, 100000);
    EXPECT_GT(overflow, 100000);
    // The absolute bound is attained (at the ties of the subnormal range); the
    // relative one is approached from below (tie just above a power of two:
    // 2^-12 / (1 + 2^-12)).
    EXPECT_EQ(worst_abs, kP16Floor);
    EXPECT_GT(worst_rel, kUP16 / (1.0L + kUP16) - kEpsLd);
    EXPECT_LE(worst_rel, kUP16);
}
