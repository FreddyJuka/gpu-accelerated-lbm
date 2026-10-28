#pragma once

// nvcc compiles a .cu file in two passes: a device pass (nvcc's own front end
// and code generator, which defines __CUDA_ARCH__) and a host pass, where the
// host code is handed to the host compiler (g++). __CUDACC__ is defined in
// both passes, whereas __CUDA_ARCH__ is defined in the device pass only.

// Host/device annotation: expands to __host__ __device__ under nvcc and to
// nothing under a plain C++ compiler, so headers compile in both. Both nvcc
// passes need the annotation (the host pass must know which functions are
// host-callable), hence the __CUDACC__ condition.
#if defined(__CUDACC__)
#define LBM_HD __host__ __device__
#else
#define LBM_HD
#endif

// Full unrolling of loops over lattice directions. Mandatory in device code:
// lattice constants are stored in local constexpr arrays, which are folded
// into immediates only when every index is a compile-time constant.
// Without unrolling, nvcc places them in local memory (verified on sm_86).
// Only the device pass expands the pragma (__CUDA_ARCH__): the host pass is
// compiled by g++, which does not recognize "#pragma unroll" and warns about
// it under -Wall. Host code has no local-memory concern.
#if defined(__CUDA_ARCH__)
#define LBM_UNROLL _Pragma("unroll")
#else
#define LBM_UNROLL
#endif
