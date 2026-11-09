# Design Decisions

This document records the architectural decisions behind `gpu-accelerated-lbm`
and the reasoning for each one. It is updated whenever a decision changes.

## Summary

| Area | Decision |
|---|---|
| Scope | 2D first (D2Q9), architecture ready for 3D (D3Q19) |
| Lattice | Template parameter |
| Collision | TRT, implemented as a policy; BGK as the special case w+ = w- |
| Precision | Template parameters `<Storage, Compute>`; FP64 for validation, FP32 + DDF-shifting for performance |
| Memory layout | Structure of Arrays (SoA), x as the fastest index |
| Propagation | Fused stream-collide kernel, pull scheme, two buffers (A-B), behind an interface |
| Lattice constants | Local `constexpr` arrays inside `LBM_HD` accessors; loops over directions fully unrolled |
| Indexing | 32-bit int, validated Q*N <= INT32_MAX |
| Node types | uint8_t flag array per node |
| Walls | Halfway bounce-back in the gather; neighbour flags read at runtime |
| Body force | Guo scheme, TRT-split, uniform force policy |
| Target hardware | NVIDIA GeForce RTX 3050 Laptop (Ampere, sm_86, 4 GB) |

## 1. Scope and lattice

The solver starts in 2D with D2Q9 to reach a validated baseline quickly
(Poiseuille, lid-driven cavity, cylinder at Re = 100). The lattice (velocity
set, weights, opposite-direction table, Q, D) is a template parameter, so the
same kernels extend to D3Q19 without rewriting.

## 2. Collision operator: TRT

The two-relaxation-time (TRT) operator relaxes the symmetric and antisymmetric
parts of the populations with rates w+ and w-:

- Kinematic viscosity: nu = cs^2 (tau+ - 1/2), with tau+ = 1/w+
- Magic parameter: Lambda = (tau+ - 1/2)(tau- - 1/2)

Lambda is a runtime input, not a compile-time constant:

- Lambda = 3/16: halfway bounce-back walls are located exactly midway between
  nodes independently of viscosity (exact Poiseuille profile).
- Lambda = 1/4: optimal stability.

BGK is recovered when w+ = w-, which provides a free cross-check. Collision is
implemented as a policy so that other operators (regularized, cumulant) can be
added later without touching the rest of the code.

## 3. Numerical precision

Precision is split into two template parameters: `Storage` (type held in
memory) and `Compute` (type used for arithmetic).

- `<double, double>`: reference configuration for validation and convergence
  studies.
- `<float, float>` with DDF-shifting: default performance configuration.
  Populations are stored shifted by their rest-state equilibrium
  (f_i - w_i for rho_0 = 1) to reduce round-off error in FP32.

Rationale: consumer Ampere GPUs run FP64 at 1/64 of FP32 throughput, and LBM
is memory-bandwidth bound, so FP32 storage is the practical production choice.
The FP64 build serves as ground truth: FP32 results must match it within a
bounded difference. FP16 storage is deferred until 3D, when memory becomes the
limiting factor.

## 4. Memory layout: SoA

Populations are stored as one contiguous array per direction:

    idx(q, x, y)    = q * N + y * nx + x,              N = nx * ny
    idx(q, x, y, z) = q * N + (z * ny + y) * nx + x,   N = nx * ny * nz

Consecutive threads map to consecutive x, so global memory accesses are
coalesced.

## 5. Propagation: fused pull, two buffers

Each thread pulls post-collision populations from its neighbours, collides, and
writes to its own node in the second buffer:

    f_i(x, t + 1) = f_i*(x - c_i, t)

Reads are misaligned and writes are aligned, which is generally cheaper on GPUs
than the push variant. Buffers are swapped every step.

Memory cost (populations only, both buffers):

| Lattice | FP32 | FP64 |
|---|---|---|
| D2Q9 | 72 B/node | 144 B/node |
| D3Q19 | 152 B/node | 304 B/node |

Propagation is hidden behind an interface so that single-buffer schemes
(AA-pattern, esoteric pull) can be added for 3D when 4 GB of VRAM becomes
limiting.

## 6. Testing and CI

- CPU reference implementation acts as the oracle for the GPU kernels.
- GitHub-hosted runners have no GPU: CI builds the CUDA code and runs CPU tests
  only. GPU tests are run locally.

## 7. Lattice constants in host and device code

nvcc rejects `static constexpr` member arrays in device code
("identifier ... is undefined in device code"), regardless of loop unrolling:
the check happens in the front end, before optimization. Lattice constants are
therefore stored as local `constexpr` arrays inside `LBM_HD` accessor functions
(`cx(q)`, `cy(q)`, `opp(q)`, `w<T>(q)`). This keeps a single source of truth
for host and device code.

Loops over lattice directions must be fully unrolled (`LBM_UNROLL`). Measured
with nvcc 12.0 on sm_86, using a kernel that computes moments of the D2Q9
weights:

| Loop | Local-memory instructions (LDL/STL) |
|---|---|
| Fully unrolled | 0 |
| Not unrolled (`#pragma unroll 1`) | 16 |

With full unrolling every index is a compile-time constant and the values are
folded into immediates. Without it, the arrays are placed in local memory.

`LBM_UNROLL` expands to `_Pragma("unroll")` only in the device pass
(`__CUDA_ARCH__`). nvcc compiles the host pass with g++, which does not
recognize the pragma and warns about it under `-Wall`; `__CUDACC__` alone is
not sufficient because it is defined in both passes.

## 8. Grid, indexing and node types

Indices are 32-bit `int`. `Grid2D` validates in its constructor, in 64-bit
arithmetic, that `Q*N <= INT32_MAX` (N = nx*ny), so no index computed by
`idx()` can overflow. With 4 GB of VRAM the limit cannot be reached: FP32 with
two buffers holds at most ~5e8 population values, below the 2^31 - 1 ~ 2.1e9
limit.

Node types are stored as a `uint8_t` flag array (`NodeType`), ~1 B per node
versus 72 B of FP32 populations per node (D2Q9).

Periodic wrap uses a conditional add/subtract instead of `%`, which is cheaper
on the GPU. It is valid for single-step neighbours only (-1 <= x <= nx).

## 9. GPU execution and verification

The CUDA kernel uses one thread per node, with 128-thread blocks along x and
one grid row per y. Device memory is owned by a move-only RAII buffer
(`DeviceBuffer`), and CUDA errors are raised as exceptions
(`LBM_CUDA_CHECK`). The CPU reference and the GPU kernel share the same node
function (`pull_stream_collide`), so they perform the same operations
in the same order. The grid's y dimension is ny, which CUDA limits (65535 on
current architectures) while `Grid2D` accepts larger values; the launcher
queries the device limit and rejects larger grids with `std::invalid_argument`.

The GPU tests are compiled with `--fmad=false` and compared bitwise against
the CPU reference. Production builds keep nvcc's default FMA contraction; a
bounded comparison with FMA enabled is planned once the error-bound helpers
move to a shared test header. Measured: with FMA contraction enabled, the
exact CPU-vs-GPU tests fail in both float and double.

## 10. Boundary conditions

Stationary walls use halfway bounce-back, applied in the gather phase of the
pull scheme. The source buffer holds post-collision populations, so when the
neighbour a population would be pulled from is a wall node, the node instead
reads its own population in the opposite direction:
`f_q = src[idx(opp(q), x, y)]`. Wall nodes are never updated; their values in
the destination buffer are left untouched. Where there are no walls the domain
stays periodic.

Neighbour flags are read at runtime from the node-type array. A precomputed
link mask is a possible later optimization if profiling shows that the flag
traffic matters.

## 11. Body force

A body force F is applied with the scheme of Guo, Zheng and Shi (2002). The
macroscopic velocity entering the equilibrium is the half-force velocity
`u = (j + F/2) / rho` (`moments_shifted_forced`), and the forcing term
`S_q = w_q [ (c_q - u)/cs2 + (c_q . u) c_q / cs2^2 ] . F` is added to the
collision with a relaxation-dependent prefactor. For TRT the source is split
into its antisymmetric part (odd under q -> opp(q), relaxed with omega_minus)
and its symmetric part (even, relaxed with omega_plus):

    out_q = f_q - omega_plus fneq_plus_q - omega_minus fneq_minus_q
            + (1 - omega_plus/2) s_sym_q + (1 - omega_minus/2) s_anti_q.

In exact arithmetic the collision conserves mass and changes the momentum by
exactly F per step. The split avoids the divisions of a generic
rate-dependent prefactor.

The force is a template policy of the collision: `TRT<T, Force>`, with
`NoForce` (default) compiling to the unforced collision (identical generated
code) and `UniformForce<T, D>` for a uniform constant force. Only the uniform
force is implemented.

## References

- I. Ginzburg, F. Verhaeghe, D. d'Humieres, "Two-relaxation-time lattice
  Boltzmann scheme: about parametrization, velocity, pressure and mixed
  boundary conditions", Commun. Comput. Phys. 3 (2008).
- M. Lehmann et al., "Accuracy and performance of the lattice Boltzmann method
  with 64-bit, 32-bit, and customized 16-bit number formats", Phys. Rev. E 106,
  015308 (2022).
- P. Bailey et al., "Accelerating lattice Boltzmann fluid flow simulations
  using graphics processors", ICPP (2009).
- U. Ghia, K. N. Ghia, C. T. Shin, "High-Re solutions for incompressible flow
  using the Navier-Stokes equations and a multigrid method", J. Comput. Phys.
  48 (1982).
- Z. Guo, C. Zheng, B. Shi, "Discrete lattice effects on the forcing term in
  the lattice Boltzmann method", Phys. Rev. E 65, 046308 (2002).
