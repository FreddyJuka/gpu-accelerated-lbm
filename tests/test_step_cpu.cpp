// Exact tests (EXPECT_EQ) of the periodic pull stream-collide step.
//
// Exact comparison suffices here because the step involves no new arithmetic:
// streaming is an exact permutation of stored values, and the step is an exact
// composition of gather + collide + store, in which the collision is the same
// code path (TRT::collide) that the reference applies to a gathered local
// array. Any indexing, direction-sign or ordering error changes bits. The
// collision physics (conservation, relaxation, fixed point) and its rounding
// error are covered by test_collision_trt.cpp.
//
// The reference gathers use explicit modular arithmetic, independent of
// Grid2D::wrap_x / wrap_y. The grid is 7 x 5 (distinct, odd) so that x/y
// mix-ups and wrap errors are visible.

#include <gtest/gtest.h>

#include <numeric>
#include <stdexcept>
#include <vector>

#include "lbm/collision/trt.hpp"
#include "lbm/cpu/step.hpp"
#include "lbm/equilibrium.hpp"
#include "lbm/grid.hpp"
#include "lbm/lattice/d2q9.hpp"

using L = lbm::D2Q9;
using Grid = lbm::Grid2D<L>;

namespace {

constexpr int NX = 7;
constexpr int NY = 5;

// Collision policy that leaves the populations unchanged (pure propagation).
struct IdentityCollision {
    template <typename Lat, typename T>
    LBM_HD void collide(T (&)[Lat::Q]) const {}
};

int mod(int a, int n) { return ((a % n) + n) % n; }

// Source index of the pull for population q arriving at (x, y), independent of
// the Grid2D wrap functions.
template <typename G>
int pull_src_index(const G& g, int q, int x, int y) {
    return g.idx(q, mod(x - L::c(q, 0), g.nx()), mod(y - L::c(q, 1), g.ny()));
}

template <typename T>
class StepCpu : public ::testing::Test {};
using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(StepCpu, PrecisionTypes);

// Distinct, exactly representable values: q*1000 + node index (< 9000 < 2^24).
template <typename T>
std::vector<T> distinct_field(const Grid& g) {
    std::vector<T> f(static_cast<std::size_t>(L::Q * g.num_nodes()));
    for (int q = 0; q < L::Q; ++q)
        for (int y = 0; y < g.ny(); ++y)
            for (int x = 0; x < g.nx(); ++x)
                f[static_cast<std::size_t>(g.idx(q, x, y))] = static_cast<T>(q * 1000 + g.node(x, y));
    return f;
}

// Non-uniform shifted populations: equilibrium of a smooth drho/u field plus
// the deterministic perturbation p_q = 1e-3 * w_q * m_q, m_q = ((3q+1) mod 7) - 3
// (as in test_moments.cpp).
template <typename T>
std::vector<T> nonuniform_field(const Grid& g) {
    std::vector<T> f(static_cast<std::size_t>(L::Q * g.num_nodes()));
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x) {
            const T drho = T(0.01) * static_cast<T>(x) - T(0.005) * static_cast<T>(y);
            const T u[2] = {T(0.02) + T(0.004) * static_cast<T>(y),
                            T(-0.01) + T(0.003) * static_cast<T>(x)};
            for (int q = 0; q < L::Q; ++q) {
                const T p = T(1e-3) * L::w<T>(q) * static_cast<T>(((3 * q + 1) % 7) - 3);
                f[static_cast<std::size_t>(g.idx(q, x, y))] =
                    lbm::equilibrium_shifted<L, T>(q, drho, u) + p;
            }
        }
    return f;
}

} // namespace

// a. One step of pure propagation equals the explicit permutation.
TYPED_TEST(StepCpu, PurePropagationOneStep) {
    using T = TypeParam;
    const Grid g(NX, NY);
    const std::vector<T> src = distinct_field<T>(g);
    std::vector<T> dst(src.size(), T(-1));
    lbm::step_periodic_cpu<L, T>(g, src.data(), dst.data(), IdentityCollision{});

    for (int q = 0; q < L::Q; ++q)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x)
                ASSERT_EQ(dst[static_cast<std::size_t>(g.idx(q, x, y))],
                          src[static_cast<std::size_t>(g.idx(q, mod(x - L::c(q, 0), NX),
                                                             mod(y - L::c(q, 1), NY)))])
                    << "q=" << q << " x=" << x << " y=" << y;

    // Explicit boundary crossings, one population per direction, with the
    // source node written out by hand. A population with c = +1 leaves through
    // the high boundary and re-enters at 0 (its source is n-1); c = -1 leaves
    // through 0 and re-enters at n-1 (its source is 0).
    auto at = [&](const std::vector<T>& v, int q, int x, int y) {
        return v[static_cast<std::size_t>(g.idx(q, x, y))];
    };
    EXPECT_EQ(at(dst, 0, 0, 0), at(src, 0, 0, 0));                  // rest
    EXPECT_EQ(at(dst, 1, 0, 2), at(src, 1, NX - 1, 2));             // (+1, 0): enters at x = 0
    EXPECT_EQ(at(dst, 2, 3, 0), at(src, 2, 3, NY - 1));             // (0, +1): enters at y = 0
    EXPECT_EQ(at(dst, 3, NX - 1, 2), at(src, 3, 0, 2));             // (-1, 0): enters at x = nx-1
    EXPECT_EQ(at(dst, 4, 3, NY - 1), at(src, 4, 3, 0));             // (0, -1): enters at y = ny-1
    EXPECT_EQ(at(dst, 5, 0, 0), at(src, 5, NX - 1, NY - 1));        // (+1,+1): corner
    EXPECT_EQ(at(dst, 6, NX - 1, 0), at(src, 6, 0, NY - 1));        // (-1,+1): corner
    EXPECT_EQ(at(dst, 7, NX - 1, NY - 1), at(src, 7, 0, 0));        // (-1,-1): corner
    EXPECT_EQ(at(dst, 8, 0, NY - 1), at(src, 8, NX - 1, 0));        // (+1,-1): corner
}

// b. Multi-step driver test. Checks that step_periodic_cpu, called repeatedly
// with alternating buffers, returns every population bitwise to its start after
// lcm(nx, ny) steps (closure of the periodic permutation, buffer alternation,
// repeated wrapping). It does NOT check the sign of the propagation direction:
// the inverse permutation also closes after lcm(nx, ny) steps. The direction is
// verified by test a (PurePropagationOneStep).
TYPED_TEST(StepCpu, DriverClosesAfterFullPeriod) {
    using T = TypeParam;
    const Grid g(NX, NY);
    const int period = std::lcm(NX, NY);
    ASSERT_EQ(period, 35);

    const std::vector<T> init = distinct_field<T>(g);
    std::vector<T> a = init;
    std::vector<T> b(init.size(), T(-1));
    for (int s = 0; s < period; ++s) {
        lbm::step_periodic_cpu<L, T>(g, a.data(), b.data(), IdentityCollision{});
        a.swap(b);
    }
    for (std::size_t i = 0; i < init.size(); ++i) ASSERT_EQ(a[i], init[i]) << "index " << i;

    // The field must have actually moved in between (guards against a no-op).
    lbm::step_periodic_cpu<L, T>(g, a.data(), b.data(), IdentityCollision{});
    EXPECT_NE(b, init);
}

// c. The step is the composition gather -> TRT::collide -> store, bitwise.
TYPED_TEST(StepCpu, CompositionWithTRT) {
    using T = TypeParam;
    const Grid g(NX, NY);
    const auto trt = lbm::TRT<T>::template from_viscosity<L>(0.01, 3.0 / 16.0);
    ASSERT_NE(trt.omega_plus(), trt.omega_minus());

    const std::vector<T> src = nonuniform_field<T>(g);
    std::vector<T> dst(src.size(), T(-1));
    lbm::step_periodic_cpu<L, T>(g, src.data(), dst.data(), trt);

    for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x) {
            T f[L::Q];
            for (int q = 0; q < L::Q; ++q) f[q] = src[static_cast<std::size_t>(pull_src_index(g, q, x, y))];
            trt.template collide<L>(f);
            for (int q = 0; q < L::Q; ++q)
                ASSERT_EQ(dst[static_cast<std::size_t>(g.idx(q, x, y))], f[q])
                    << "q=" << q << " x=" << x << " y=" << y;
        }
}

// d. A uniform state maps every node to the same collided populations.
TYPED_TEST(StepCpu, UniformState) {
    using T = TypeParam;
    const Grid g(NX, NY);
    const auto trt = lbm::TRT<T>::template from_viscosity<L>(0.01, 3.0 / 16.0);

    T f0[L::Q];
    const T u[2] = {T(0.03), T(-0.02)};
    for (int q = 0; q < L::Q; ++q)
        f0[q] = lbm::equilibrium_shifted<L, T>(q, T(0.01), u) +
                T(1e-3) * L::w<T>(q) * static_cast<T>(((3 * q + 1) % 7) - 3);

    std::vector<T> src(static_cast<std::size_t>(L::Q * g.num_nodes()));
    for (int q = 0; q < L::Q; ++q)
        for (int n = 0; n < g.num_nodes(); ++n) src[static_cast<std::size_t>(q * g.num_nodes() + n)] = f0[q];

    T expected[L::Q];
    for (int q = 0; q < L::Q; ++q) expected[q] = f0[q];
    trt.template collide<L>(expected);

    std::vector<T> dst(src.size(), T(-1));
    lbm::step_periodic_cpu<L, T>(g, src.data(), dst.data(), trt);
    for (int q = 0; q < L::Q; ++q)
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x)
                ASSERT_EQ(dst[static_cast<std::size_t>(g.idx(q, x, y))], expected[q])
                    << "q=" << q << " x=" << x << " y=" << y;
}

// e. Invalid arguments are rejected.
TYPED_TEST(StepCpu, RejectsInvalidArguments) {
    using T = TypeParam;
    const Grid g(NX, NY);
    std::vector<T> a(static_cast<std::size_t>(L::Q * g.num_nodes()), T(0));
    std::vector<T> b(a.size(), T(0));
    const IdentityCollision id;
    auto step = [&](const T* src, T* dst) { lbm::step_periodic_cpu<L, T>(g, src, dst, id); };

    EXPECT_THROW(step(a.data(), a.data()), std::invalid_argument);
    EXPECT_THROW(step(nullptr, b.data()), std::invalid_argument);
    EXPECT_THROW(step(a.data(), nullptr), std::invalid_argument);
    EXPECT_THROW(step(nullptr, nullptr), std::invalid_argument);
    EXPECT_NO_THROW(step(a.data(), b.data()));
}
