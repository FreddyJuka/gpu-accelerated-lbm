#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "lbm/grid.hpp"
#include "lbm/lattice/d2q9.hpp"

using L = lbm::D2Q9;
using Grid = lbm::Grid2D<L>;

static_assert(std::is_same<std::underlying_type<lbm::NodeType>::type, std::uint8_t>::value,
              "NodeType must be stored as uint8_t");
static_assert(sizeof(lbm::NodeType) == 1, "NodeType must occupy one byte");
static_assert(std::is_trivially_copyable<Grid>::value, "Grid2D must be trivially copyable");

namespace {

constexpr std::int64_t kIntMax = std::numeric_limits<std::int32_t>::max();
constexpr int kMaxNodes = static_cast<int>(kIntMax / L::Q);

// Returns the message of the std::invalid_argument thrown by Grid(nx, ny),
// or an empty string if nothing is thrown.
std::string rejection_message(int nx, int ny) {
    try {
        Grid g(nx, ny);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

} // namespace

TEST(Grid2D, AcceptsValidSizes) {
    EXPECT_NO_THROW(Grid(1, 1));
    EXPECT_NO_THROW(Grid(5, 3));
    EXPECT_NO_THROW(Grid(1024, 1024));

    const Grid g(5, 3);
    EXPECT_EQ(g.nx(), 5);
    EXPECT_EQ(g.ny(), 3);
    EXPECT_EQ(g.num_nodes(), 15);
}

TEST(Grid2D, RejectsNonPositiveSizesNamingQuantity) {
    for (int bad : {0, -1}) {
        const std::string mx = rejection_message(bad, 4);
        EXPECT_NE(mx.find("nx"), std::string::npos) << mx;
        EXPECT_NE(mx.find(std::to_string(bad)), std::string::npos) << mx;

        const std::string my = rejection_message(4, bad);
        EXPECT_NE(my.find("ny"), std::string::npos) << my;
        EXPECT_NE(my.find(std::to_string(bad)), std::string::npos) << my;
    }
}

TEST(Grid2D, OverflowLimitAtBoundary) {
    ASSERT_EQ(L::Q, 9);
    // The largest accepted node count satisfies Q*N <= INT32_MAX, the next
    // one does not.
    ASSERT_LE(static_cast<std::int64_t>(L::Q) * kMaxNodes, kIntMax);
    ASSERT_GT(static_cast<std::int64_t>(L::Q) * (static_cast<std::int64_t>(kMaxNodes) + 1), kIntMax);

    EXPECT_NO_THROW(Grid(kMaxNodes, 1));
    EXPECT_NO_THROW(Grid(1, kMaxNodes));
    EXPECT_THROW(Grid(kMaxNodes + 1, 1), std::invalid_argument);
    EXPECT_THROW(Grid(1, kMaxNodes + 1), std::invalid_argument);

    // Q*20000*20000 = 3.6e9 > INT32_MAX; the product nx*ny alone (4e8) fits
    // in int, so only the Q factor triggers the rejection.
    const std::string msg = rejection_message(20000, 20000);
    EXPECT_FALSE(msg.empty());
    EXPECT_NE(msg.find("Q*nx*ny"), std::string::npos) << msg;
    EXPECT_NE(msg.find(std::to_string(static_cast<std::int64_t>(L::Q) * 20000 * 20000)),
              std::string::npos)
        << msg;
}

TEST(Grid2D, LargestGridLastIndex) {
    const Grid g(kMaxNodes, 1);
    const std::int64_t N = g.num_nodes();
    EXPECT_EQ(N, kMaxNodes);
    const std::int64_t last = g.idx(L::Q - 1, g.nx() - 1, g.ny() - 1);
    EXPECT_EQ(last, static_cast<std::int64_t>(L::Q) * N - 1);
    EXPECT_LE(last, kIntMax);

    const Grid h(1, kMaxNodes);
    EXPECT_EQ(static_cast<std::int64_t>(h.idx(L::Q - 1, h.nx() - 1, h.ny() - 1)),
              static_cast<std::int64_t>(L::Q) * h.num_nodes() - 1);
}

TEST(Grid2D, IndexingIsBijection) {
    const Grid g(5, 3);
    const int total = L::Q * g.num_nodes();
    std::vector<int> hits(static_cast<std::size_t>(total), 0);
    for (int q = 0; q < L::Q; ++q)
        for (int y = 0; y < g.ny(); ++y)
            for (int x = 0; x < g.nx(); ++x) {
                const int i = g.idx(q, x, y);
                ASSERT_GE(i, 0);
                ASSERT_LT(i, total);
                ++hits[static_cast<std::size_t>(i)];
            }
    for (int i = 0; i < total; ++i) EXPECT_EQ(hits[static_cast<std::size_t>(i)], 1) << "index " << i;
}

TEST(Grid2D, Strides) {
    const Grid g(7, 4);
    for (int q = 0; q < L::Q - 1; ++q)
        for (int y = 0; y < g.ny() - 1; ++y)
            for (int x = 0; x < g.nx() - 1; ++x) {
                EXPECT_EQ(g.idx(q, x + 1, y) - g.idx(q, x, y), 1);
                EXPECT_EQ(g.idx(q, x, y + 1) - g.idx(q, x, y), g.nx());
                EXPECT_EQ(g.idx(q + 1, x, y) - g.idx(q, x, y), g.num_nodes());
                EXPECT_EQ(g.node(x, y), y * g.nx() + x);
                EXPECT_EQ(g.idx(0, x, y), g.node(x, y));
            }
}

TEST(Grid2D, PeriodicWrap) {
    for (int n : {1, 2, 5}) {
        const Grid gx(n, 3);
        EXPECT_EQ(gx.wrap_x(-1), n - 1);
        EXPECT_EQ(gx.wrap_x(n), 0);
        for (int x = 0; x < n; ++x) EXPECT_EQ(gx.wrap_x(x), x);

        const Grid gy(3, n);
        EXPECT_EQ(gy.wrap_y(-1), n - 1);
        EXPECT_EQ(gy.wrap_y(n), 0);
        for (int y = 0; y < n; ++y) EXPECT_EQ(gy.wrap_y(y), y);
    }
}
