#pragma once

// Reference implementation of the P16 storage format (lbm/storage.hpp) for the
// tests: decode and encode written from the definition of the format (1 sign
// bit, 4 exponent bits, 11 mantissa bits) with std::ldexp on integers and
// comparisons of exact long double values. It shares no code with
// lbm::StorageConversion<lbm::P16, float>.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "support/error_bounds.hpp"

namespace lbmtest {

inline constexpr std::uint32_t kP16Invalid = 0x7FFFu;  // magnitude bits of the invalid pattern
inline constexpr int kP16Magnitudes = 0x8000;          // number of 15-bit magnitude patterns

// Value of the magnitude pattern k = (e << 11) | m as a number: 2^(e - 15)
// (1 + m/2048) = (2048 + m) 2^(e - 26) for e >= 1, m 2^-25 for e == 0. Exact:
// a 12-bit integer times a power of two. Defined for k = 0x7FFF as well (the
// value 2 - 2^-11 the invalid pattern would have as a number), which the
// rounding reference needs as the upper neighbour of the largest finite value.
inline ld p16_magnitude_value(std::uint32_t k) {
    const int e = static_cast<int>((k >> 11) & 0xFu);
    const int m = static_cast<int>(k & 0x7FFu);
    return e != 0 ? std::ldexp(static_cast<ld>(2048 + m), e - 26) : std::ldexp(static_cast<ld>(m), -25);
}

// Expected pattern of store(x) by exact comparison. For finite x with
// a = |x| < magnitude_value(0x7FFF), k is the largest magnitude pattern with
// value(k) <= a (the values increase with k, checked in test_storage.cpp); a lies in
// [value(k), value(k + 1)) and is rounded to the nearer of the two, to the even
// pattern in a tie. The comparison is 2a against value(k) + value(k + 1): 2a is
// exact (a power-of-two scaling of a binary32 value) and the sum of two
// adjacent values has at most 13 significant bits, so no rounding occurs and
// no subtraction is needed. The result 0x7FFF is the invalid pattern, which is
// also what a >= value(0x7FFF), inf and NaN map to.
inline std::uint16_t p16_reference_store(float x, const std::vector<ld>& values) {
    const std::uint32_t sign = std::signbit(x) ? 0x8000u : 0u;
    if (std::isnan(x) || std::isinf(x)) return static_cast<std::uint16_t>(sign | kP16Invalid);
    const ld a = std::abs(static_cast<ld>(x));
    if (a >= values[kP16Invalid]) return static_cast<std::uint16_t>(sign | kP16Invalid);
    const std::uint32_t k = static_cast<std::uint32_t>(
        std::upper_bound(values.begin(), values.end(), a) - values.begin() - 1);
    const ld twice = 2.0L * a;
    const ld mid2 = values[k] + values[k + 1];
    std::uint32_t r = k;
    if (twice > mid2 || (twice == mid2 && (k & 1u) != 0u)) r = k + 1;
    return static_cast<std::uint16_t>(sign | r);
}

inline std::vector<ld> p16_magnitude_values() {
    std::vector<ld> v(static_cast<std::size_t>(kP16Magnitudes));
    for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(kP16Magnitudes); ++k) v[k] = p16_magnitude_value(k);
    return v;
}

// Value of a 16-bit pattern as a binary32 number (exact: 12 significant bits):
// the sign bit applied to the magnitude value; NaN for the invalid pattern.
inline float p16_reference_load(std::uint16_t pattern) {
    const std::uint32_t k = pattern & 0x7FFFu;
    if (k == kP16Invalid) return std::numeric_limits<float>::quiet_NaN();
    const float a = static_cast<float>(p16_magnitude_value(k));
    return (pattern & 0x8000u) != 0u ? -a : a;
}

}  // namespace lbmtest
