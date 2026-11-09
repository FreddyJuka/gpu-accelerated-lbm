// Exact tests (EXPECT_EQ) of halfway bounce-back in the pull stream-collide
// step.
//
// Exact comparison suffices: bounce-back moves stored values without new
// arithmetic, and the collision is the same code path (TRT::collide) applied
// by the reference to a gathered local array. The reference gathers use
// explicit modular arithmetic, independent of Grid2D::wrap_x / wrap_y, and
// read the flags directly.
//
// Geometries (grid 9 x 5, non-square):
//   G1: channel, walls on the full rows y = 0 and y = ny-1.
//   G2: one wall node at (4, 2) in the interior, surrounded by fluid. Its
//       eight neighbours cover every D2Q9 direction; the four diagonal
//       neighbours touch it only diagonally.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "lbm/collision/trt.hpp"
#include "lbm/cpu/step.hpp"
#include "lbm/equilibrium.hpp"
#include "lbm/grid.hpp"
#include "lbm/lattice/d2q9.hpp"

using L = lbm::D2Q9;
using Grid = lbm::Grid2D<L>;
using lbm::NodeType;

namespace {

constexpr int NX = 9;
constexpr int NY = 5;
constexpr int WX = 4; // G2 wall node
constexpr int WY = 2;

// Collision policy that leaves the populations unchanged (pure propagation).
struct IdentityCollision {
    template <typename Lat, typename T>
    LBM_HD void collide(T (&)[Lat::Q]) const {}
};

enum class Geometry { Channel, SingleWall };

const char* name(Geometry geo) { return geo == Geometry::Channel ? "G1 channel" : "G2 single wall"; }

std::vector<NodeType> make_flags(const Grid& g, Geometry geo) {
    std::vector<NodeType> flags(static_cast<std::size_t>(g.num_nodes()), NodeType::Fluid);
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x) {
            const bool wall = (geo == Geometry::Channel) ? (y == 0 || y == g.ny() - 1)
                                                         : (x == WX && y == WY);
            if (wall) flags[static_cast<std::size_t>(g.node(x, y))] = NodeType::Wall;
        }
    return flags;
}

int mod(int a, int n) { return ((a % n) + n) % n; }

bool is_wall(const Grid& g, const std::vector<NodeType>& flags, int x, int y) {
    return flags[static_cast<std::size_t>(g.node(mod(x, g.nx()), mod(y, g.ny())))] == NodeType::Wall;
}

// Index in src of the population that the reference gathers into (q, x, y).
int ref_src_index(const Grid& g, const std::vector<NodeType>& flags, int q, int x, int y) {
    const int xs = mod(x - L::c(q, 0), g.nx());
    const int ys = mod(y - L::c(q, 1), g.ny());
    return is_wall(g, flags, xs, ys) ? g.idx(L::opp(q), x, y) : g.idx(q, xs, ys);
}

template <typename T>
class BounceBack : public ::testing::Test {};
using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(BounceBack, PrecisionTypes);

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

const Geometry kGeometries[] = {Geometry::Channel, Geometry::SingleWall};

} // namespace

// a. Reflection: with identity collision, every fluid population is either
// the node's own opposite population (wall source) or the plain pull.
TYPED_TEST(BounceBack, Reflection) {
    using T = TypeParam;
    const Grid g(NX, NY);
    for (const Geometry geo : kGeometries) {
        SCOPED_TRACE(name(geo));
        const std::vector<NodeType> flags = make_flags(g, geo);
        const std::vector<T> src = distinct_field<T>(g);
        std::vector<T> dst(src.size(), T(-1));
        lbm::step_cpu<L, T>(g, flags.data(), src.data(), dst.data(), IdentityCollision{});

        int reflected = 0;
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x) {
                if (flags[static_cast<std::size_t>(g.node(x, y))] == NodeType::Wall) continue;
                for (int q = 0; q < L::Q; ++q) {
                    const int xs = mod(x - L::c(q, 0), NX);
                    const int ys = mod(y - L::c(q, 1), NY);
                    const T expected = is_wall(g, flags, xs, ys)
                                           ? src[static_cast<std::size_t>(g.idx(L::opp(q), x, y))]
                                           : src[static_cast<std::size_t>(g.idx(q, xs, ys))];
                    ASSERT_EQ(dst[static_cast<std::size_t>(g.idx(q, x, y))], expected)
                        << "q=" << q << " x=" << x << " y=" << y;
                    if (is_wall(g, flags, xs, ys)) ++reflected;
                }
            }
        // Guard against vacuous passes: G1 has 2 rows x 9 nodes x 3 wall-facing
        // directions each side, G2 has 8 wall-facing links (one per neighbour).
        EXPECT_EQ(reflected, geo == Geometry::Channel ? 2 * NX * 3 : 8);
    }
}

// b. Wall nodes are never updated.
TYPED_TEST(BounceBack, WallNodesUntouched) {
    using T = TypeParam;
    const Grid g(NX, NY);
    const T sentinel = T(-12345);
    for (const Geometry geo : kGeometries) {
        SCOPED_TRACE(name(geo));
        const std::vector<NodeType> flags = make_flags(g, geo);
        const std::vector<T> src = nonuniform_field<T>(g);
        const auto trt = lbm::TRT<T>::template from_viscosity<L>(0.01, 3.0 / 16.0);
        std::vector<T> dst(src.size(), sentinel);
        lbm::step_cpu<L, T>(g, flags.data(), src.data(), dst.data(), trt);

        int walls = 0;
        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x) {
                const bool wall = flags[static_cast<std::size_t>(g.node(x, y))] == NodeType::Wall;
                walls += wall ? 1 : 0;
                for (int q = 0; q < L::Q; ++q) {
                    const T v = dst[static_cast<std::size_t>(g.idx(q, x, y))];
                    if (wall)
                        ASSERT_EQ(v, sentinel) << "q=" << q << " x=" << x << " y=" << y;
                    else
                        ASSERT_NE(v, sentinel) << "q=" << q << " x=" << x << " y=" << y;
                }
            }
        EXPECT_EQ(walls, geo == Geometry::Channel ? 2 * NX : 1);
    }
}

// c. With identity collision, one step permutes the fluid populations.
TYPED_TEST(BounceBack, PermutesFluidPopulations) {
    using T = TypeParam;
    const Grid g(NX, NY);
    for (const Geometry geo : kGeometries) {
        SCOPED_TRACE(name(geo));
        const std::vector<NodeType> flags = make_flags(g, geo);
        const std::vector<T> src = distinct_field<T>(g);
        std::vector<T> dst(src.size(), T(-1));
        lbm::step_cpu<L, T>(g, flags.data(), src.data(), dst.data(), IdentityCollision{});

        std::vector<T> before, after;
        for (int q = 0; q < L::Q; ++q)
            for (int y = 0; y < NY; ++y)
                for (int x = 0; x < NX; ++x) {
                    if (flags[static_cast<std::size_t>(g.node(x, y))] == NodeType::Wall) continue;
                    before.push_back(src[static_cast<std::size_t>(g.idx(q, x, y))]);
                    after.push_back(dst[static_cast<std::size_t>(g.idx(q, x, y))]);
                }
        ASSERT_EQ(before.size(), after.size());
        std::sort(before.begin(), before.end());
        std::sort(after.begin(), after.end());
        // The values are pairwise distinct, so equal sorted sequences mean no
        // population was lost or duplicated.
        ASSERT_TRUE(std::adjacent_find(before.begin(), before.end()) == before.end());
        EXPECT_EQ(before, after);
        // The step must actually have moved something.
        EXPECT_NE(std::vector<T>(dst.begin(), dst.end()), src);
    }
}

// d. The step is the composition gather (with bounce-back) -> TRT::collide ->
// store, bitwise, for every fluid node.
TYPED_TEST(BounceBack, CompositionWithTRT) {
    using T = TypeParam;
    const Grid g(NX, NY);
    const auto trt = lbm::TRT<T>::template from_viscosity<L>(0.01, 3.0 / 16.0);
    ASSERT_NE(trt.omega_plus(), trt.omega_minus());

    for (const Geometry geo : kGeometries) {
        SCOPED_TRACE(name(geo));
        const std::vector<NodeType> flags = make_flags(g, geo);
        const std::vector<T> src = nonuniform_field<T>(g);
        std::vector<T> dst(src.size(), T(-1));
        lbm::step_cpu<L, T>(g, flags.data(), src.data(), dst.data(), trt);

        for (int y = 0; y < NY; ++y)
            for (int x = 0; x < NX; ++x) {
                if (flags[static_cast<std::size_t>(g.node(x, y))] == NodeType::Wall) continue;
                T f[L::Q];
                for (int q = 0; q < L::Q; ++q)
                    f[q] = src[static_cast<std::size_t>(ref_src_index(g, flags, q, x, y))];
                trt.template collide<L>(f);
                for (int q = 0; q < L::Q; ++q)
                    ASSERT_EQ(dst[static_cast<std::size_t>(g.idx(q, x, y))], f[q])
                        << "q=" << q << " x=" << x << " y=" << y;
            }
    }
}

// e. A null flags pointer is rejected.
TYPED_TEST(BounceBack, RejectsNullFlags) {
    using T = TypeParam;
    const Grid g(NX, NY);
    const std::vector<NodeType> flags = make_flags(g, Geometry::Channel);
    std::vector<T> a(static_cast<std::size_t>(L::Q * g.num_nodes()), T(0));
    std::vector<T> b(a.size(), T(0));
    const IdentityCollision id;

    auto step = [&](const NodeType* fl) { lbm::step_cpu<L, T>(g, fl, a.data(), b.data(), id); };

    EXPECT_THROW(step(nullptr), std::invalid_argument);
    EXPECT_NO_THROW(step(flags.data()));
}
