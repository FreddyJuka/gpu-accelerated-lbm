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
