#pragma once

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
// Extension to a 16-bit storage format (later phase). The format is a
// distinct, trivially copyable class type that wraps the bit pattern, e.g.
//
//   struct P16 { std::uint16_t bits; };
//
// and not a bare std::uint16_t: the trait is selected by the type, so the
// format needs a type of its own (several 16-bit formats can then coexist),
// and a wrapper has no implicit arithmetic conversions, so a population cannot
// be used as a number without going through load. The buffers are then arrays
// of P16 (2 bytes per population, same SoA indexing) and
// StorageConversion<P16, float> implements
//
//   load:   decode sign, exponent and mantissa fields of `bits` with integer
//           shifts and masks, assemble the binary32 bit pattern, reinterpret;
//   store:  reinterpret the binary32 value as its bit pattern, then encode
//           with integer shifts and masks, including the rounding (round to
//           nearest, ties to even, done on the integer mantissa), the
//           subnormal range and saturation at the largest finite value.
//
// Everything between the two reinterpretations is integer arithmetic on
// fixed-width unsigned types, which is exact and identical on every platform,
// so CPU and GPU agree bitwise by construction, independent of FMA
// contraction, of F16C / hardware half-float support and of cuda_fp16.h
// (which CPU-only builds do not have). No interface change is needed for
// this: the step code is already written against load and store only.
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
