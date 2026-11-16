# Design Decisions

This document records the architectural decisions behind `gpu-accelerated-lbm`
and the reasoning for each one. It is updated whenever a decision changes.

## Summary

| Area | Decision |
|---|---|
| Scope | 2D first (D2Q9); the collision-side code is lattice-generic, grid, node function and kernels are 2D and need 3D counterparts |
| Lattice | Template parameter |
| Collision | TRT, implemented as a policy; BGK as the special case w+ = w- |
| Precision | Template parameters `<Storage, Compute>` joined by `StorageConversion`: double/double (validation), float/float with DDF-shifting (performance), float/double (mixed), experimental 16-bit P16/float |
| Memory layout | Structure of Arrays (SoA), x as the fastest index |
| Propagation | Fused stream-collide node function, pull scheme, two buffers (A-B); no separate propagation interface |
| Lattice constants | Local `constexpr` arrays inside `LBM_HD` accessors; loops over directions fully unrolled |
| Indexing | 32-bit int, validated Q*N <= INT32_MAX |
| Node types | uint8_t flag array per node |
| Walls | Halfway bounce-back in the gather; neighbour flags read at runtime |
| Body force | Guo scheme, TRT-split, uniform force policy |
| GPU mapping | One thread per node over the flattened node index, 128-thread 1D blocks |
| Target hardware | NVIDIA GeForce RTX 3050 Laptop (Ampere, sm_86, 4 GB) |

## 1. Scope and lattice

The solver starts in 2D with D2Q9 to reach a validated baseline quickly
(Poiseuille now; lid-driven cavity and cylinder at Re = 100 are on the
roadmap). The lattice (velocity set, weights, opposite-direction table, Q, D)
is a template parameter.

What is lattice-generic and what is not:

- Generic: shifted equilibrium, moments, the Guo source, the TRT collision.
  They use only `L::D`, `L::Q`, `L::c(q, d)`, `L::opp(q)`, `L::w<T>(q)`,
  `L::cs2<T>()` and `L::inv_cs2<T>()`.
- 2D only: `Grid2D` (it has a `static_assert(L::D == 2)`), the node function
  `pull_stream_collide` (it takes `Grid2D` and `(x, y)`), `step_cpu`,
  `step_kernel` and the launch configuration. D3Q19 needs 3D counterparts of
  these; it is not a matter of instantiating the existing kernels with another
  lattice.
- There is no separate propagation interface: `step_cpu` and `step_kernel`
  call `pull_stream_collide` directly.

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

Precision is split into two template parameters: `Storage` (type of one
population in memory) and `Compute` (type used for arithmetic). The two meet
only in `StorageConversion<Storage, Compute>` (`include/lbm/storage.hpp`):

    static constexpr bool supported;
    LBM_HD static Compute load(Storage s);    // after the gather
    LBM_HD static Storage store(Compute c);   // before the write

The node function gathers `Storage` values, converts them with `load`,
collides in `Compute` (the compute type is declared by the collision policy as
`compute_type`; a policy without one computes in the storage type) and
converts the result with `store`. Streaming and bounce-back move `Storage`
values and never look inside them. The stored quantity is always the shifted
population f~_i = f_i - w_i; the equilibrium and the moments work with the
density deviation drho = rho - 1, and the full f_i is not formed.

Implemented combinations:

| Storage | Compute | load / store | Role |
|---|---|---|---|
| double | double | identity | reference for validation and convergence studies |
| float | float | identity | default performance configuration |
| float | double | exact widening / one IEEE rounding (mixed precision) | implemented and tested |
| P16 | float | exact / round to nearest, integer operations | experimental |

Any other pair is rejected at compile time (`static_assert` on `supported`);
there is deliberately no generic `static_cast` fallback, so every combination
is an explicit decision with its own tests. Each conversion is a single IEEE
operation or integer arithmetic, so it is bitwise identical on CPU and GPU
(tests: `StepGpuMixed`, `StepGpuP16`, `StepGpuP16.ConversionMatchesHostBitwise`).

Rationale: consumer Ampere GPUs run FP64 at 1/64 of FP32 throughput, and LBM
is memory-bandwidth bound, so FP32 storage is the practical production choice.
Storing populations shifted by their rest-state equilibrium (f_i - w_i for
rho_0 = 1) reduces round-off error in FP32 (Lehmann et al. 2022). The FP64
build serves as ground truth.

### 3.1 Float accuracy

The float/float configuration is validated on Poiseuille flow only where the
derived tolerance resolves the slip (section 12). Measured with
`lbm_poiseuille --compute float` (GPU, 24 cases from rest, error against
`u_a + du` relative to u_max = 0.01): 2.4e-7 to 3.4e-4. The slip is resolved
in 10 of the 12 cases with Lambda = 1/4; it is not at nu = 0.01 with H = 32
(error 1.03 times the slip) and H = 64 (1.81 times), and only marginally at
nu = 1/6 with H = 64 (0.60). max |u_y| is below 2.7e-7 and the drift of the
mean density below 4.1e-5 (largest in the longest run, 316 000 steps).

### 3.2 P16: experimental 16-bit storage with float compute

Format (`lbm::P16`, 1 sign, 4 exponent, 11 mantissa bits; the "FP16C" layout
of Lehmann et al.):

    bit 15      sign
    bits 14-11  exponent field e
    bits 10-0   mantissa field m
    e >= 1:  (-1)^s 2^(e-15) (1 + m/2048)    2^-14 <= |v| <= 2 - 2^-10, 12 significant bits
    e == 0:  (-1)^s m 2^-25                  subnormal, constant spacing 2^-25
    e == 15, m == 2047: invalid pattern (loads as NaN)

- `load` is exact. `store` rounds to nearest, ties to even, on the integer
  significand; no floating-point operation is involved. The error is at most
  2^-12 |x| in the normal range and 2^-26 below it. Magnitudes up to 2^-26 are
  stored as zero (sign kept).
- Overflow is not saturated: a finite |x| beyond the largest finite value
  (rounding included) and +-inf and NaN are stored as the invalid pattern,
  which loads as NaN. A population outside (-2, 2) is a failed simulation, and
  a NaN spreads through the following collisions where a clamped value would
  continue as a plausible number. The rounding rule and the invalid pattern
  are choices made in this project, not taken from the paper.
- An all-zero buffer is the rest state.
- Layout choice: the shifted populations of the validation flows span about 24
  binades. In a prototype comparison (scratch program, not in the repository)
  against binary16 scaled by 2^-15 (1-5-10), this layout gave the smaller
  Poiseuille error in 23 of 24 cases, by a factor 1.8 in the geometric mean.
  The price is the absolute floor: below 2^-14 populations have fewer than 12
  significant bits, and flows with a velocity scale below about 1e-3 lose
  relative accuracy with this format.
- Tests (`tests/test_storage.cpp`): decoding of all 65536 patterns and rounding
  of about 4.2 million floats plus the neighbourhood of every representable
  value and tie, both against an independent reference
  (`tests/support/p16_reference.hpp`) that shares no code with the conversion;
  monotonicity; special values; the rounding-error bounds.

**Memory.** Populations take 36 B per node (D2Q9, two buffers), 37 B per node
update with the flag byte. 8192 x 8192 (67 million nodes, 2368 MiB) fits in
4 GB and runs; float storage needs 4672 MiB there and stops at 4096 x 4096
(1168 MiB).

**Speed** (GPU, large grids, channel, float/float against P16/float; section 13
has the conditions):

| | expected from traffic (73 / 37) | thermally limited (4096^2) | cool GPU (2048^2) |
|---|---|---|---|
| unforced | 1.97 | 1.28 to 1.29 | 1.46 |
| forced | 1.97 | 1.13 to 1.14 | 1.31 |

P16 reaches 65 % (unforced) and 58 % (forced) of the copy-kernel bandwidth at
4096^2, against 100 % for float. The reason is instruction count, not traffic:
the P16 step executes 753 SASS instructions per node unforced (833 forced)
against 371 (451) for float, because the conversion is integer bit
manipulation (21 instructions per conversion, 18 conversions per node). The
product MLUPS x instructions per node is the same for the unforced and the
forced kernel (2.76e12 and 2.74e12 instructions per second on the cool GPU),
which is the signature of an instruction-bound kernel; this is inferred from
instruction counts, since hardware counters are not available (section 13). At
that rate the traffic ratio would need about 10 instructions per conversion
instead of 21. On small grids P16 is slower than float (128^2: 0.77 to 0.83,
launch-bound); the first run, from which this comes, was disturbed and is
indicative only.

**Accuracy limitation.** A force-driven flow started from rest does not reach
the steady state with P16 storage: every run ends at an exact fixed point of
the rounded scheme (final change 0), short of the steady state. Against
`u_a + du`, relative to u_max, over the 24 Poiseuille cases of
`lbm_poiseuille --compute float --storage p16`: 3.7e-4 (0.04 %) in the best case
(H = 8, nu = 0.5) to 0.55 in the worst (H = 64, nu = 0.01), between 580 and
33 000 times the float error. The slip is resolved by measurement in 2 of 12
cases with Lambda = 1/4. max |u_y| and the mean-density drift stay at the
level of float (below 6e-6 and 7.5e-5); the damage is in u_x. Over the 20
cases whose error is below 0.2 u_max the error follows

    err / u_max = c * 2^-12 * H^2 / nu,    c between 0.012 and 0.033,

the four cases with nu = 0.01 and H >= 32 saturate. This is an empirical fit,
not a derivation. Since F_x = 8 nu u_max / H^2, the error is inversely
proportional to the force, and independent of u_max because the format is
relative. P16 is therefore accurate for such flows only when 2^-12 H^2 / nu is
small: narrow, viscous channels (0.04 to 0.06 % at H = 8, nu = 0.5; 1 % at
H = 32, nu = 0.5; 4 % at H = 8, nu = 0.01). Flows driven by boundaries or
started from a developed field were not measured beyond a periodic
Taylor-Green vortex in a development program (kinetic energy within 3e-4 of
float after 4000 steps at u0 = 0.05); that program is not part of the
repository, so the result cannot be reproduced from it.

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

| Lattice | FP32 (also float storage with double compute) | FP64 | 16-bit (P16) |
|---|---|---|---|
| D2Q9 | 72 B/node | 144 B/node | 36 B/node |
| D3Q19 | 152 B/node | 304 B/node | 76 B/node |

There is no propagation interface: the CPU step and the CUDA kernel call the
node function `pull_stream_collide` directly. A single-buffer scheme
(AA-pattern, esoteric pull) would halve the population memory but changes the
data-layout contract of the step (post-collision populations in the source
buffer, which the Poiseuille measurement relies on), so it needs a new node
function. On this GPU no speed-up is expected from it: float is already at the
copy bandwidth and double is bound by the FP64 pipe (section 13). It is on the
roadmap.

## 6. Testing and CI

- The CPU reference implementation acts as the oracle for the GPU kernels. The
  CPU-vs-GPU tests compare bitwise, after 1 step and after 100 steps, on
  7 x 5 and 200 x 37 grids with channel and single-wall flags, for float and
  double, unforced and forced, for float storage with double compute, and for
  P16 storage with float compute. Tall narrow grids (1 x 70000 and 3 x 70000,
  5 steps, unforced float and double) are compared as well.
- Tolerances are upper bounds derived from error bounds and operation counts
  (u = eps/2). The shared constants K_* and their derivations are in
  `tests/support/error_bounds.hpp`; every factor is at most 16, enforced with
  `static_assert`; scales are sums of magnitudes; references are in long
  double. Where the exact result is known bitwise, the tests use equality.
  The exception is the Poiseuille tolerance (section 12), which rests on a
  physical estimate.
- Forced collision (`tests/test_force.cpp`): zero force reproduces the unforced
  collision bitwise; parity of the source parts; moments of the source and the
  mass and momentum change of the collision in long double; composition from
  its parts (pins the expression of the code, does not check the formula); the
  second moment of the collision including the symmetric source, from an
  independent expression and with distinct rates (the symmetric Guo term is
  tested here, not by Poiseuille); and the accuracy of the float and double
  forced collision against the same library code evaluated in long double
  (this checks rounding, not the formula).
- The exact and bitwise tests assume IEEE binary32/binary64 with no excess
  precision (`FLT_EVAL_METHOD == 0`), checked at compile time in
  `tests/support/fp_environment.hpp`, and no contraction of `a*b+c` into FMA on
  the host. The latter cannot be checked at compile time: `tests/CMakeLists.txt`
  passes `-ffp-contract=off` to every test target (`-Xcompiler=-ffp-contract=off`
  for CUDA sources), because GCC otherwise contracts on hosts with FMA and the
  tests fail. On the device the CPU-vs-GPU tests use `--fmad=false`.
- Currently 208 tests, of which 57 carry the label `gpu` and need a GPU.

CI (`.github/workflows/ci.yml`, `ubuntu-24.04`) has three jobs:

| Job | Configuration | Runs |
|---|---|---|
| `cuda` | Release, `LBM_WARNINGS_AS_ERRORS=ON`, `LBM_ENABLE_CUDA=ON`, sm_86; nvcc 12.0 from the Ubuntu package `nvidia-cuda-toolkit` | builds everything, including the GPU tests; `ctest -LE gpu` |
| `cpu-fma` | Release, `LBM_ENABLE_CUDA=OFF`, `LBM_WARNINGS_AS_ERRORS=ON`, `-march=x86-64-v3` | all tests, with FMA instructions available to the compiler |
| `cpu-sanitizers` | Debug, `LBM_ENABLE_CUDA=OFF`, `-fsanitize=address,undefined`, `UBSAN_OPTIONS=halt_on_error=1` | all tests |

GitHub-hosted runners have no GPU, so no kernel is executed in CI: the badge
covers the CUDA compilation and the CPU tests, not GPU execution. Discovery of
the GPU tests needs no GPU (checked with `CUDA_VISIBLE_DEVICES=""`), so the
full build succeeds on the runner. The GPU tests are run locally.

Build configuration: `LBM_ENABLE_CUDA` (default ON) makes the CPU reference and
its tests buildable without nvcc; with it ON and no CUDA compiler,
configuration fails instead of falling back. GoogleTest v1.18.0 is fetched with
a pinned SHA-256 hash. C++ sources are compiled with `-Wall -Wextra -Wpedantic`;
the host pass of CUDA sources gets `-Wall -Wextra` only.

## 7. Lattice constants in host and device code

nvcc rejects `static constexpr` member arrays in device code
("identifier ... is undefined in device code"), regardless of loop unrolling:
the check happens in the front end, before optimization. Lattice constants are
therefore stored as local `constexpr` arrays inside `LBM_HD` accessor functions
(`cx(q)`, `cy(q)`, `opp(q)`, `w<T>(q)`). This keeps a single source of truth
for host and device code.

Loops over lattice directions must be fully unrolled (`LBM_UNROLL`); without
it the constant arrays are placed in local memory. During development a kernel
that computes moments of the D2Q9 weights, compiled with nvcc 12.0 for sm_86,
showed 0 local-memory instructions (LDL/STL) with full unrolling and 16 with
`#pragma unroll 1`. That kernel is not in the repository, so this comparison
cannot be reproduced from it. What the repository reproduces is the check on
the production code: `tests/device_compile_check.cu` instantiates the device
code (equilibrium, moments, TRT unforced and forced, grid, pull step, storage
conversions, for float and double and for the mixed and P16 combinations), and

    cuobjdump -sass build/tests/CMakeFiles/lbm_device_compile_check.dir/device_compile_check.cu.o | grep -cE 'LDL|STL'
    cuobjdump -res-usage build/tests/CMakeFiles/lbm_device_compile_check.dir/device_compile_check.cu.o

report 0 LDL/STL instructions and `STACK:0 LOCAL:0` for every kernel of the
Release build, with 39 to 64 registers per thread. The pull-step kernels of the
performance report use 40 registers (float; 46 unforced and 62 forced for
double after the flattened mapping).

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

The CUDA kernel uses one thread per node over the flattened node index
`node = blockIdx.x * blockDim.x + threadIdx.x = y * nx + x` (the SoA node
order), in one-dimensional blocks of 128 threads; threads past the last node
return. Consecutive threads handle consecutive nodes across row boundaries, so
every warp is full whatever nx is, and coalescing is unchanged because the SoA
index `q * N + node` is contiguous in the node index. This replaced one
128-thread block row per y: with that mapping a narrow grid left most lanes of
each warp idle, and the cost of a warp instruction, in particular on the FP64
pipe, does not depend on the number of active lanes.

Measured gain of the flattened mapping against the old one, in an interleaved
comparison (both binaries run alternately so that they see the same GPU state;
ratio of medians, second round; taken in the lower power state of the first
performance study, section 13.2, so only the ratios are used):

| Grid | Double, unforced / forced | Float, unforced / forced |
|---|---|---|
| 4 x 4096 (the validation channel has nx = 4) | 7.5x / 7.6x | 2.6x / 2.7x |
| 16 x 4096 | 1.97x / 1.97x | 1.10x / 1.14x |
| 100 x 4096 | 1.28x / 1.27x | within the spread |
| Square grids 128^2 to 2048^2 | 0.99 to 1.00 | 0.99 to 1.00 |

Square grids are unchanged within the spread. Other variants measured and not
adopted: `__restrict__` (no change), 64- and 256-thread blocks (same as 128
within the spread), `__launch_bounds__(128, 12)` (double forced spills, 36
LDL/STL, and is 6 % slower). The block size stays 128 (full theoretical
occupancy for 40-register kernels).

There is no limit on ny: the grid is one-dimensional, and the only size limit
is that of `Grid2D`, `Q*nx*ny <= INT32_MAX`, which gives at most 2^24 blocks.
(An earlier version rejected ny above 65535, the CUDA limit of the y grid
dimension; `StepGpu.TallNarrowGridsMatchCpu` now checks such grids.) The
launcher throws `std::invalid_argument` for null pointers (flags included) and
for `src == dst`, and `std::runtime_error` if the launch fails. Device memory
is owned by a move-only RAII buffer (`DeviceBuffer`), and CUDA errors are
raised as exceptions (`LBM_CUDA_CHECK`). The CPU reference and the GPU kernel
share the same node function (`pull_stream_collide`), so they perform the same
operations in the same order.

The GPU tests are compiled with `--fmad=false` and compared bitwise against
the CPU reference. Production builds keep nvcc's default FMA contraction.
Measured: with FMA contraction enabled, the exact CPU-vs-GPU tests fail in both
float and double. A bounded comparison with FMA enabled is not implemented yet
(roadmap); the error-bound helpers it needs are now in
`tests/support/error_bounds.hpp`.

## 10. Boundary conditions

Stationary walls use halfway bounce-back, applied in the gather phase of the
pull scheme. The source buffer holds post-collision populations, so when the
neighbour a population would be pulled from is a wall node, the node instead
reads its own population in the opposite direction:
`f_q = src[idx(opp(q), x, y)]`. Wall nodes are never updated; their values in
the destination buffer are left untouched. Where there are no walls the domain
stays periodic.

The reflection is applied to shifted populations, `f~_q <- f~_opp(q)`. This
equals the unshifted rule `f_q <- f_opp(q)` only because `w_q = w_opp(q)`
(tested by `D2Q9Moments.WeightsAreSymmetric` in `tests/test_lattice_d2q9.cpp`).
The TRT collision itself does not need this symmetry, since it splits
`fneq = f~ - f~eq = f - feq`. Any future boundary rule that mixes directions
must respect the same condition.

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
`NoForce` (default) for the unforced collision and `UniformForce<T, D>` for a
uniform constant force. Only the uniform force is implemented. `NoForce` keeps
the size of the unforced object (checked with `static_assert`); that it
generates the same code as a collision without the policy was compared once
during development and is not tested.

## 12. Validation: Poiseuille flow

The body-force-driven channel (periodic in x, walls at the rows y = 0 and
y = ny - 1, halfway bounce-back, so the walls sit at y = 1/2 and
y = ny - 3/2 and the width is H) is the validation case of the boundary and
force implementation. With TRT and halfway bounce-back the steady solution is
the analytical parabola `u_a(y) = F_x/(2 nu) (y - 1/2)(ny - 3/2 - y)` shifted
by a uniform slip

    du = F_x/(2 nu) * (16 Lambda - 3) / 12,

which vanishes at Lambda = 3/16 (equivalently H_eff^2 = H^2 - 1 + (16/3)
Lambda). Source: Ginzburg, Verhaeghe, d'Humieres, Commun. Comput. Phys. 3(2)
(2008), Eq. (2.29) with Table 1 (bounce-back, delta_q = 1/2), listed in the
References.

Measurements with `apps/poiseuille` (double precision, GPU, u_max = 0.01,
steady state to a change of 1e-14 u_max per 1000 steps): for Lambda = 3/16 and
Lambda = 1/4, H = 8, 16, 32, 64 and nu = 0.01, 1/6, 0.5, the numerical profile
equals `u_a + du` at round-off level (residual at most 2.7e-12 u_max, the
largest values at the largest H and smallest nu, i.e. the longest runs; at
Lambda = 3/16 the error against the plain parabola is at the same level). The
fitted uniform offset agrees with the predicted `du` to better than 1e-8
relative, and the spread of the offset across the channel is at round-off
level. The formula is thus confirmed for these two values; Lambda = 1/12 is
covered by the tests only.

The velocity is measured on the collision input: the stored populations are
post-collision, and the collision adds F to the momentum, so
`moments_shifted_forced` on the stored populations would be off by about F.
The measurement runs the gather of the production node function (identity
collision) and applies `moments_shifted_forced` to the result.

The tests (`tests/test_poiseuille.cpp` on the CPU, `tests/gpu/test_poiseuille_gpu.cu`
on the GPU, shared code in `tests/poiseuille_common.hpp`) cover H = 8, 16,
nu = 1/6, 0.5 and Lambda = 3/16, 1/4, 1/12 (12 cases per backend) and check
u_x against `u_a + du`, and u_y and drho against zero. They run in three
precisions, with these scopes:

- double/double: all 12 cases;
- float/float: only the cases whose derived tolerance is below the slip, i.e.
  H = 8 with both viscosities and H = 16 with nu = 0.5 (9 cases). The
  derived tolerance grows as eps (H + 1)^2 / nu while the slip decreases as
  1/H^2; at H = 16 with nu = 1/6 it is about the slip itself, so such a case
  could not validate the slip formula. The restriction follows from the derived
  tolerance, not from the measured errors, and the same assertion as in double
  (`|du| > tol_x`) fails a float case that stops resolving the slip;
- P16/float: all 12 cases, with weaker claims. The tolerance contains the
  rounding of the 16-bit store (each step moves a population by at most
  2^-12 |s|, or 2^-26 below the normal range) and is 0.4 % to 4.5 % of u_max.
  It is below the slip only for H = 8 with nu = 0.5, and only there (with
  Lambda = 1/4 and 1/12; at Lambda = 3/16 the slip is zero) is the slip
  resolution claimed. The other cases check u_x, u_y and drho within the
  derived bounds without claiming to resolve the slip (a result that ignored
  the slip would pass); what they assert instead is that the tolerance is
  below the smallest reference velocity in the channel, so that a stalled or
  grossly wrong profile fails. The measured P16 errors (section 3.2) are
  larger than the slip in most cases; the tests do not cover wider channels or
  nu = 0.01 for P16. The reference uses the nu and Lambda effectively
stored in the collision object, which are asserted to agree with the nominal
values within their derived rounding bounds. The tolerance is
`delta_step/(1 - r) + residual_stop + ref_round`: `delta_step` is the round-off
one step introduces, from operation counts (the mass and momentum bounds of the
force tests, evaluated on the gathered populations of every node at every
check); `r` is the decay factor per step of the slowest diffusive mode,
`exp(-nu pi^2/(H + 1)^2)`; `residual_stop` is `change r^K/(1 - r^K)` for the
measured final change over the last K = 1000 steps; `ref_round` is the
rounding of the long double reference. The tolerance is therefore an
estimate resting on a stated physical assumption, not a proven upper bound.
The assumption is
stated explicitly: `r` is a physical estimate of the slowest hydrodynamic
mode, not a proven contraction bound for the full (non-normal) LBM operator;
the tolerance also assumes that round-off enters through the conserved
moments with the bounds above, that the same `r` applies to u_y and drho, and
that the exact-arithmetic steady state is `u_a + du`, u_y = 0, drho = 0. The
GPU test target keeps nvcc's default FMA contraction; that the derivation also
holds for fused arithmetic is asserted in the test comments, not argued.

The one proven part is the mean of drho: streaming and bounce-back permute
populations (exact in floating point) and the collision conserves mass in
exact arithmetic, so the mean density can change only by the per-step
round-off of the collision, which accumulates linearly (`n` steps times the
per-node mass bound, plus the rounding of the measurement). That bound needs no
physical assumption (it is loose: with P16 storage it is about 3.5e-3 against a
measured drift of 3e-7 to 1.4e-5). The fluctuation of drho around the mean
uses the decay assumption.

The app takes `--compute double|float` and `--storage same|float|p16`
(`--storage p16` requires float compute); the default is double/double, the
configuration of the measurements above. Its output has one line per case with
the error against `u_a`, against `u_a + du` (`err_slip`), `slip`, `max|u_y|`,
`max|drho|` and `mean_drho`. With float compute the stopping threshold is the
same number of eps as in double. The float and P16 results of section 3 are
reproduced with

    ./build/apps/lbm_poiseuille --compute float
    ./build/apps/lbm_poiseuille --compute float --storage p16

These tests do not exercise the symmetric part of the Guo source. In this
flow u_y = 0 and F_y = 0, so its c_x c_y moment, u_x F_y + u_y F_x, vanishes,
and removing the term leaves the steady profile unchanged at round-off level
(checked by mutation). The symmetric term is tested separately in
`tests/test_force.cpp`: `ForceCollide.SecondMomentIncludesSymmetricSource`
checks, in long double and with distinct rates, that the second moment of the
collision equals `-omega_plus Pi_neq + (1 - omega_plus/2)(u_a F_b + u_b F_a)`
computed from an independent expression, so a wrong prefactor or a missing
term changes the left side by more than the derived bound (the test asserts
this precondition). The composition test (`CompositionFromParts`) only pins
the expression of the code bitwise.

## 13. Performance

### 13.1 Benchmark app and metrics

`apps/benchmark` (target `lbm_benchmark`, built with CUDA) calls the production
launcher `lbm::step_gpu`, with two buffers swapped every step and one flag byte
per node, and times with CUDA events. Per case it runs a warm-up, then `reps`
repetitions of `steps` steps (with `--steps auto` the smallest count, at least
10, for which a repetition lasts at least `--rep-ms`), without host
synchronisation inside the loop. It reports median, minimum and maximum time
per step, the spread (max - min) / median, and:

- MLUPS = nx ny steps / time / 1e6, all nodes counted (wall nodes included);
- bytes per node update = `2 * Q * sizeof(Storage) + 1`: Q populations read and
  Q written, plus one flag byte (the neighbour flag reads hit cache lines
  already fetched and are not counted): 73 (float), 145 (double), 37 (P16), 73
  (float storage with double compute);
- effective bandwidth = bytes per node update x nodes x steps / time;
- the fraction of the theoretical peak (memory clock x 2 x bus width / 8 =
  192.0 GB/s for this GPU) and the fraction of the **copy-kernel bandwidth**,
  the reference used as "100 %": a grid-stride device-to-device copy with
  16-byte elements over two 256 MiB buffers, measured in the same run.

Options include `--sizes` (N or NXxNY), `--storage float,double,p16,mixed|all`
(configurations interleaved size by size so that they see the same clocks),
`--precision`, `--geometry periodic|channel`, `--forcing`, `--reps`,
`--rep-ms`. Geometries are an all-fluid periodic domain and a channel with
walls at y = 0 and y = ny - 1; the forced cases use the Guo source.

### 13.2 Conditions and limits of the measurements

The results below were measured with the FP16 phase, on AC power with the
Windows power plan "Performance": performance state P0 in every loaded sample,
memory clock 6000 MHz (copy kernel 183.1 to 183.6 GB/s, 95.3 to 95.6 % of the
peak), power draw 53 to 94 W (mean 70 W), temperature 65 C at the start and
86 to 87 C after about 40 s of load, SM clock 1905 to 1942 MHz below 86 C and
1560 to 1890 MHz once the thermal slowdown flag is set. Another Windows
application held 1.2 to 3.0 GB of the 4 GB of device memory during the session;
with 3.0 GB held the large grids showed spreads of 20 to 60 %, so they were
measured again with 1.2 GB held and that run is tabulated. The float kernel is
memory-bound and does not react to the SM clock; the P16 kernel is
instruction-bound and does, so its results depend on the thermal state and both
states are given.

Other limits, stated plainly:

- One machine, one driver (WSL2); the GPU is a laptop part whose power and
  thermal state vary between sessions.
- Nsight Compute cannot collect hardware counters here (`ERR_NVGPUCTRPERM`,
  enforced by the Windows driver) and Nsight Systems records no kernel data
  under WSL2. The statements "memory-bound", "FP64-bound" and
  "instruction-bound" are therefore derived from SASS instruction counts
  (`cuobjdump -sass`), resource usage (`cuobjdump -res-usage`), the copy-kernel
  bandwidth and rooflines, not from counters.
- The first performance study ran in a lower power state (P3, memory clock
  5469 MHz, 35 W power cap, copy kernel 167.5 GB/s). Its absolute numbers are
  not comparable with the table below. It is used here only for relative
  comparisons between binaries run interleaved in the same session
  (section 9), for instruction counts and for the FP64-pipe fraction.
- The launch micro-benchmark behind the 10 us figure is a scratch program
  that is not in the repository.

### 13.3 Results

Large grids, `lbm_benchmark --storage float,p16 --sizes 1024,2048,4096,8192
--reps 7 --rep-ms 500`, median of 7 repetitions, channel geometry (the periodic
geometry differs by a few percent at most), thermally limited state:

| Grid | float/float unforced / forced MLUPS | P16/float unforced / forced MLUPS |
|---|---|---|
| 4096^2 | 2520 / 2519 (100.5 % of copy) | 3230 / 2852 (65.3 % / 57.7 %) |
| 8192^2 | does not fit | 3247 / 2856 (65.6 % / 57.7 %) |

Three of the four 2048^2 float cases were disturbed (spread 19 to 28 %) and
are not used. From 1024^2 upward the undisturbed float value is 2500 to 2520
MLUPS.

All four configurations, `lbm_benchmark --storage all --sizes 2048 --geometry
channel --reps 7 --rep-ms 500`, cool GPU (67 C at the start, 84 C after the P16
cases):

| Storage / compute | Bytes per node update | Unforced MLUPS (% of copy) | Forced MLUPS (% of copy) |
|---|---|---|---|
| float / float | 73 | 2516 (100.0) | 2516 (100.0) |
| P16 / float | 37 | 3662 (73.8) | 3292 (66.3) |
| double / double | 145 | 354 (27.9) | 237 (18.7) |
| float / double | 73 | 316 (12.6) | 219 (8.7) |

Reading:

- **Float** runs at about 100 % of the copy-kernel bandwidth from 1024^2
  upward (73 B per node update): on large grids it cannot get faster without
  moving fewer bytes.
- **Double** is bound by the FP64 pipe, not by memory (28 % of the copy
  bandwidth). The GA107 executes FP64 at 1/64 of the FP32 rate; the node
  function has 164 FP64 instructions per node unforced and 237 forced (SASS),
  and the measured rate was 87 to 89 % of the pipe limit computed from these
  counts at the SM clock observed (first study, section 13.2; the remainder is
  the reciprocal, compares and integer work). The forced/unforced ratio of 1.45 equals the instruction
  ratio 237 / 164.
- **Float storage with double compute** is slower than double/double at the
  same grid (316 against 354 MLUPS unforced); the reports do not analyse why.
- **P16 / float** is instruction-bound (section 3.2): 1.28 to 1.29 times float
  unforced and 1.13 to 1.14 forced at 4096^2 when thermally limited, 1.46 and
  1.31 on the cool GPU, against 1.97 from traffic alone. Its benefit on this GPU
  is capacity (8192^2).
- **Small grids** are bound by the cost of a kernel launch under WSL2,
  measured at 9.5 to 10.4 us per launch of an empty kernel (0.02 us for
  `cudaGetDevice` or `cudaDeviceGetAttribute`). A float kernel on a grid up to
  256 x 256 is shorter than that. In the first AC run, which was
  disturbed and is indicative only, float reached 1900 to 1984 MLUPS at
  128^2 and 1913 to 2224 MLUPS at 256^2 and 512^2 (75 to 88 % of copy);
  P16 was slower than float at 128^2 and ahead from 512^2 on.
- **Narrow grids**: see section 9 for the effect of the thread mapping.

### 13.4 Reproduction

    cmake -S . -B build -DLBM_WARNINGS_AS_ERRORS=ON
    cmake --build build -j
    ./build/apps/lbm_benchmark --storage float,p16 --sizes 1024,2048,4096,8192 --reps 7 --rep-ms 500
    ./build/apps/lbm_benchmark --storage all --sizes 2048 --geometry channel --reps 7 --rep-ms 500
    ./build/apps/lbm_benchmark --reps 7 --rep-ms 500 --sizes 4x1024,4x4096,16x4096,32x4096,64x4096,100x4096,200x4096,1024x64,4096x16

Record the clocks next to a run:

    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,pstate,temperature.gpu,clocks_throttle_reasons.active --format=csv -lms 1000

### 13.5 Not implemented

Candidates from the profile; each changes the interface, the traffic or the
arithmetic and needs a design decision (estimates from the performance study,
not measurements):

- CUDA graphs or a multi-step kernel for launch-bound grids (a stepper object
  instead of a free function); expected to help grids below 512^2, not larger
  ones.
- Dropping the c = 0 terms in the generic loops (6 of the 18 (q, d) pairs
  after unrolling): 164 to 152 FP64 instructions unforced, 237 to 213 forced,
  expected 7 % and 10 % for double; bitwise identical for finite inputs but
  the operation sequence changes, so the operation-count bounds of the tests
  would have to be re-audited.
- A hardware binary16 conversion for P16: it would cut the instruction count
  but means the binary16 layout, the less accurate of the two formats compared
  (section 3.2).
- Single-buffer propagation (section 5): memory, not speed.

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
