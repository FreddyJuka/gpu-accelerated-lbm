#pragma once

// Host/device annotation: expands to __host__ __device__ under nvcc and to
// nothing under a plain C++ compiler, so headers compile in both.
#if defined(__CUDACC__)
#define LBM_HD __host__ __device__
#else
#define LBM_HD
#endif

// Full unrolling of loops over lattice directions. Mandatory in device code:
// lattice constants are stored in local constexpr arrays, which are folded
// into immediates only when every index is a compile-time constant.
// Without unrolling, nvcc places them in local memory (verified on sm_86).
#if defined(__CUDACC__)
#define LBM_UNROLL _Pragma("unroll")
#else
#define LBM_UNROLL
#endif
