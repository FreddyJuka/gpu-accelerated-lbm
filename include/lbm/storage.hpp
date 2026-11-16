#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "lbm/config.hpp"

namespace lbm {

// Storage/Compute precision split.
//
// Storage is the type of one population as it lies in the population buffers;
// Compute is the type every arithmetic operation of the step is carried out
// in. StorageConversion<Storage, Compute> is the only place where the two
// meet:
//
//   static constexpr bool supported;         true in every specialization
//   LBM_HD static Compute load(Storage s);   after the gather
//   LBM_HD static Storage store(Compute c);  before the write
//
// The node function (pull_stream_collide) gathers Storage values, converts
// each with load, collides in Compute and converts each result with store.
// Streaming and halfway bounce-back only move Storage values and never look
// inside them. The stored quantity is always the shifted population
// f~_q = f_q - w_q; a conversion maps numbers to numbers and never adds or
// removes the shift.
//
// The trait is keyed on the pair of types, and both functions are stateless,
// take and return by value and are static, so a specialization is free to
// implement them in any way that is valid in host and device code. The
// primary template is the "unsupported" case: a combination is available only
// if a specialization exists, and pull_stream_collide rejects every other one
// at compile time (static_assert on `supported`). There is deliberately no
// generic static_cast fallback: each combination is an explicit decision with
// its own tests.
//
// Specializations provided here:
//
//   <float, float>, <double, double>   identity. load and store return their
//       argument, so the Storage == Compute configurations perform exactly the
//       operations they performed before the split.
//   <float, double>   load is the exact widening binary32 -> binary64; store
//       is the IEEE conversion binary64 -> binary32 in the current rounding
//       mode (round to nearest, ties to even, on the host by default and
//       always on the device). Both are single IEEE operations, hence bitwise
//       identical on CPU and GPU.
//
//   <P16, float>   16-bit storage with binary32 compute; see P16 below. load is
//       exact, store rounds to nearest (ties to even) to 12 significant bits.
//       Both are integer operations on the bit patterns, hence bitwise
//       identical on CPU and GPU, independent of FMA contraction, of F16C /
//       hardware half-float support and of cuda_fp16.h (which CPU-only builds
//       do not have).
//
// A 16-bit format is a distinct, trivially copyable class type that wraps the
// bit pattern, and not a bare std::uint16_t: the trait is selected by the
// type, so the format needs a type of its own (several 16-bit formats can
// coexist), and a wrapper has no implicit arithmetic conversions, so a
// population cannot be used as a number without going through load.
//
// Requirements on a specialization: Storage is trivially copyable (buffers
// are copied with memcpy / cudaMemcpy and passed to kernels as raw pointers),
// load and store are LBM_HD and free of side effects, and
// store(load(s)) == s for every finite value s that store can produce.
template <typename Storage, typename Compute>
struct StorageConversion {
    static constexpr bool supported = false;
};

namespace detail {

// Storage == Compute: no conversion.
template <typename T>
struct IdentityStorageConversion {
    static constexpr bool supported = true;
    LBM_HD static constexpr T load(T s) { return s; }
    LBM_HD static constexpr T store(T c) { return c; }
};

} // namespace detail

template <>
struct StorageConversion<float, float> : detail::IdentityStorageConversion<float> {};

template <>
struct StorageConversion<double, double> : detail::IdentityStorageConversion<double> {};

// binary32 storage, binary64 compute.
template <>
struct StorageConversion<float, double> {
    static constexpr bool supported = true;
    // Exact: every binary32 value is a binary64 value.
    LBM_HD static constexpr double load(float s) { return static_cast<double>(s); }
    // One correctly rounded IEEE conversion.
    LBM_HD static constexpr float store(double c) { return static_cast<float>(c); }
};

// ---------------------------------------------------------------------------
// P16: 16-bit storage format for shifted populations
// ---------------------------------------------------------------------------
// A sign-magnitude binary floating-point format with 1 sign bit, 4 exponent
// bits and 11 mantissa bits:
//
//   bit 15      sign s
//   bits 14-11  exponent field e (0..15)
//   bits 10-0   mantissa field m (0..2047)
//
//   e >= 1:  value = (-1)^s 2^(e - 15) (1 + m / 2048)      normal, 12 significant bits
//   e == 0:  value = (-1)^s 2^(-14) (m / 2048) = m 2^-25   subnormal (m == 0: zero)
//   e == 15, m == 2047: not a number (the "invalid" pattern, see below).
//
// Normal values cover 2^-14 <= |v| <= 2 - 2^-10 (6.1e-5 to 1.9990) with the
// relative spacing 2^-11; below 2^-14 the spacing is the constant 2^-25
// (3.0e-8), down to zero. The magnitude of a value increases with the 15-bit
// pattern (e, m), so the patterns of one sign are ordered like the values.
//
// Why this layout. The shifted populations f~_q = f_q - w_q of a weakly
// compressible flow are bounded well below 2 (f_q >= 0 gives f~_q >= -4/9,
// and the rest population f~_0 = 4/9 drho at rest reaches 2 only at
// rho = 5.5), and their size is w_q |drho| and 3 w_q |u|: about 3e-3 for the
// axis directions at |u| = 0.01 and up to 0.06 at |u| = 0.15. Fifteen binades
// below 2 hold these velocity-carrying populations as normal numbers with 12
// significant bits, one more than IEEE binary16 (1-5-10) scaled to the same
// range, whose fifth exponent bit extends the normal range down to 2e-9
// instead. The populations that small exist (the terms of second order in
// |u|: a third of the populations of the Poiseuille validation cases at
// u_max = 0.01 lie below 2^-14), but the error of the flow is governed by the
// relative rounding of the large ones: in a prototype comparison of the two
// layouts on the 24 Poiseuille cases of apps/poiseuille, 1-4-11 had the
// smaller velocity error in 23, by a factor 1.8 in the geometric mean. The
// price is the absolute floor: a magnitude below 2^-26 (1.5e-8) is stored as
// zero, and populations below 2^-14 carry fewer than 12 significant bits.
// Flows whose velocity scale is below about 1e-3 therefore lose relative
// accuracy with this format. The layout is the 1-4-11 split studied as "FP16C"
// by Lehmann et al. (Phys. Rev. E 106, 015308, 2022); the rounding (ties to
// even) and the invalid pattern in place of saturation are choices made here.
//
// load (P16 -> float) is exact: every P16 value is a binary32 value.
//
// Both conversions are written without data-dependent branches (the device
// code of the step has the same four branches as with float storage), so
// their cost does not depend on the values: a field at rest, where every
// population is zero, costs the same as a developed flow.
//
// store (float -> P16) rounds to the nearest P16 value, ties to the pattern
// with an even mantissa field, on the integer significand:
//
//   2^-14 <= |x| <= 2 - 2^-10 + 2^-12:
//                               |load(store(x)) - x| <= 2^-12 |x|
//   |x| < 2^-14:                |load(store(x)) - x| <= 2^-26  (half the
//                               subnormal spacing; relative error unbounded)
//   |x| <= 2^-26:               stored as zero (2^-26 is a tie, the even
//                               pattern is zero); float subnormals included
//   zero:                       the sign is kept in both directions, and an
//                               underflow keeps the sign of x (-0.0f <-> 0x8000)
//   overflow:                   a finite x that rounds beyond the largest
//                               finite value, i.e. |x| > 2 - 2^-10 + 2^-12
//                               (that value is a tie and rounds down to the
//                               even pattern), is stored as the invalid pattern
//   +-inf, NaN:                 stored as the invalid pattern (sign bit kept).
//
// The invalid pattern (0x7FFF, 0xFFFF) loads as a quiet NaN. The format has no
// infinities. Overflow is deliberately not saturated: a population outside
// (-2, 2) is a failed simulation (negative f_q or rho > 4), and a NaN spreads
// through the following collisions and is seen, whereas a clamped value would
// continue as a plausible number. This is what float storage does as well
// (overflow -> inf -> NaN in the next collision).
//
// store(load(s)) == s for every pattern s (for the two invalid patterns up to
// the sign bit, which a NaN is not guaranteed to keep on every platform).
// An all-zero buffer is the rest state (+0 in every population).
struct P16 {
    std::uint16_t bits;
};

static_assert(std::is_trivially_copyable_v<P16>);
static_assert(sizeof(P16) == 2);

namespace detail {

// Reinterpretation of a binary32 value as its bit pattern and back. memcpy of
// four bytes is folded into a register move by g++ and nvcc.
LBM_HD inline std::uint32_t float_bits(float x) {
    std::uint32_t b;
    std::memcpy(&b, &x, sizeof b);
    return b;
}

LBM_HD inline float float_from_bits(std::uint32_t b) {
    float x;
    std::memcpy(&x, &b, sizeof x);
    return x;
}

// Number of leading zeros of m within an 11-bit field (0 for m >= 2^10, 10 for
// m == 1), for 0 < m < 2^11; for other m the result is unspecified but the
// call is valid, and load does not use it. An exact integer function, computed
// with the count-leading-zeros instruction of the target: m is moved to the
// top of the word with a guard bit below the field, so that the argument is
// never zero (__builtin_clz(0) is undefined). The portable cascade is the
// definition, for host compilers without the builtin.
LBM_HD inline std::uint32_t p16_leading_zeros(std::uint32_t m) {
#if defined(__CUDA_ARCH__)
    return static_cast<std::uint32_t>(__clz(static_cast<int>((m << 21) | 0x100000u)));
#elif defined(__GNUC__)
    return static_cast<std::uint32_t>(__builtin_clz((m << 21) | 0x100000u));
#else
    std::uint32_t t = m;
    std::uint32_t sh = 0u;
    if (t < 0x8u) {
        t <<= 8;
        sh += 8u;
    }
    if (t < 0x80u) {
        t <<= 4;
        sh += 4u;
    }
    if (t < 0x200u) {
        t <<= 2;
        sh += 2u;
    }
    if (t < 0x400u) {
        t <<= 1;
        sh += 1u;
    }
    return sh;
#endif
}

} // namespace detail

template <>
struct StorageConversion<P16, float> {
    static constexpr bool supported = true;

    // Exact. binary32 fields: exponent field = unbiased exponent + 127,
    // 23 mantissa bits. Written without early returns so that the compilers
    // can turn the three cases into selects (the device code has no branch).
    LBM_HD static float load(P16 s) {
        const std::uint32_t b = s.bits;
        const std::uint32_t sign = (b & 0x8000u) << 16;
        const std::uint32_t k = b & 0x7FFFu;  // (e << 11) | m
        // Normal (e >= 1): 2^(e - 15) (1 + m / 2^11) has the exponent field
        // e - 15 + 127 = e + 112 and the mantissa m << 12. Shifting k left by
        // 12 puts e at bit 23 and m at bits 22..12 in one step.
        const std::uint32_t normal = (k << 12) + (112u << 23);
        // Subnormal (e == 0, k == m != 0): m 2^-25. Shift the leading one of m
        // to bit 10 (t in [2^10, 2^11), m = t 2^-sh, sh in 0..10); then
        // m 2^-25 = (t / 2^10) 2^(-15 - sh), exponent field 112 - sh, and the
        // 10 bits of t below the leading one are the top mantissa bits.
        const std::uint32_t sh = detail::p16_leading_zeros(k);
        const std::uint32_t t = k << sh;
        const std::uint32_t subnormal = ((112u - sh) << 23) | ((t & 0x3FFu) << 13);
        std::uint32_t magnitude = (k >= 0x800u) ? normal : subnormal;
        if (k == 0u) magnitude = 0u;                // zero (t and sh are unused)
        if (k == 0x7FFFu) magnitude = 0x7FC00000u;  // invalid: quiet NaN
        return detail::float_from_bits(sign | magnitude);
    }

    // Round to nearest, ties to even, on the 24-bit significand. No branches.
    LBM_HD static P16 store(float c) {
        const std::uint32_t b = detail::float_bits(c);
        const std::uint32_t sign = (b >> 16) & 0x8000u;
        // Exponent field the value would have as a normal P16 number.
        const int e = static_cast<int>((b >> 23) & 0xFFu) - 112;
        // |c| = sig 2^(e - 15 - 23) with the 24-bit significand sig (leading
        // one included). A normal result (e >= 1) keeps the top 12 bits
        // (shift 12); the leading one then lands on bit 11 and adds 1 to the
        // exponent field, hence the base (e - 1) << 11. A subnormal result
        // (e <= 0) is a multiple of 2^-25: shift 13 - e, base 0. With shift 24
        // the quotient is 0 and the remainder is compared with 2^23 (|c|
        // against the tie 2^-26). From shift 25 on, |c| < 2^-26 rounds to
        // zero; the shift is limited to 25, where the quotient is 0 and the
        // remainder sig < 2^24 is below the half 2^24, so the result is 0 as
        // required. binary32 zeros and subnormals (exponent field 0, e = -112)
        // end there too, so their missing leading one is never used.
        const bool normal = e >= 1;
        const int wide = (13 - e < 25) ? 13 - e : 25;
        const std::uint32_t shift = normal ? 12u : static_cast<std::uint32_t>(wide);
        const std::uint32_t base = normal ? static_cast<std::uint32_t>(e - 1) << 11 : 0u;
        const std::uint32_t sig = (b & 0x7FFFFFu) | 0x800000u;
        std::uint32_t h = base + (sig >> shift);
        const std::uint32_t rem = sig & ((1u << shift) - 1u);
        const std::uint32_t half = 1u << (shift - 1u);
        // Round up if rem > half, or if rem == half and h is odd: for integers
        // this is rem + (h & 1) > half. A carry out of the mantissa field
        // increments the exponent field, which is the correct result (the next
        // binade, or the smallest normal number from the largest subnormal one).
        h += (rem + (h & 1u) > half) ? 1u : 0u;
        // Everything not representable ends above 0x7FFE and becomes the
        // invalid pattern: the rounded value 2 - 2^-11 (h == 0x7FFF), a carry
        // beyond it, and |c| >= 2, inf and NaN, for which e >= 16 and
        // base >= 15 << 11, so that h >= 0x8000 (e <= 143, no wrap-around).
        if (h > 0x7FFFu) h = 0x7FFFu;
        return P16{static_cast<std::uint16_t>(sign | h)};
    }
};

namespace detail {

template <typename Collision, typename Storage, typename = void>
struct CollisionComputeType {
    using type = Storage;
};

template <typename Collision, typename Storage>
struct CollisionComputeType<Collision, Storage, std::void_t<typename Collision::compute_type>> {
    using type = typename Collision::compute_type;
};

} // namespace detail

// Compute type of a step that runs the collision policy Collision on buffers
// of Storage: the policy's member type `compute_type` (TRT<Compute, Force>
// declares it). A policy without that member is taken to be generic in the
// type of the populations (its collide deduces it); it then computes in
// Storage.
template <typename Collision, typename Storage>
using collision_compute_t = typename detail::CollisionComputeType<Collision, Storage>::type;

} // namespace lbm
