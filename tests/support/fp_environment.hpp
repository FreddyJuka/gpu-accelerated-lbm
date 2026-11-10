#pragma once

// Floating-point environment assumed by the exact (EXPECT_EQ) and bitwise
// tests. Those tests compare results that are equal only if every operation
// is a single correctly rounded IEEE operation, evaluated in the stated type
// and in the source order:
//
//   - float and double are IEEE 754 binary32 and binary64 (checked below);
//   - no excess precision: FLT_EVAL_METHOD == 0, so a float expression is
//     evaluated in float and a double expression in double (checked below);
//   - no contraction of a*b+c into FMA on the host. This cannot be checked
//     portably at compile time; tests/CMakeLists.txt passes
//     -ffp-contract=off (and --fmad=false for the CPU-vs-GPU comparison on the
//     device) to every test target.
//
// Include this header from every test file with exact or bitwise comparisons.

#include <cfloat>
#include <limits>

static_assert(std::numeric_limits<float>::is_iec559,
              "exact tests require IEEE 754 binary32 float");
static_assert(std::numeric_limits<double>::is_iec559,
              "exact tests require IEEE 754 binary64 double");
static_assert(FLT_EVAL_METHOD == 0,
              "exact tests require FLT_EVAL_METHOD == 0 (no excess precision)");
