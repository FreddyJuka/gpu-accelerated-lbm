#pragma once

// Shared test inputs: the state set, the deterministic perturbation, population
// fields and geometries used by several test files. Nothing here depends on
// the rounding-error analysis (see support/error_bounds.hpp for that).
//
// Everything is in namespace lbmtest. A test file brings it in with
// `using namespace lbmtest;`; it does not define its own L or Grid.

#include <cstddef>
#include <vector>

#include "lbm/config.hpp"
#include "lbm/equilibrium.hpp"
#include "lbm/grid.hpp"
#include "lbm/lattice/d2q9.hpp"

namespace lbmtest {

using L = lbm::D2Q9;
using Grid = lbm::Grid2D<L>;

constexpr int delta(int a, int b) { return a == b ? 1 : 0; }

inline int mod(int a, int n) { return ((a % n) + n) % n; }

// Collision policy that leaves the populations unchanged (pure propagation).
struct IdentityCollision {
    template <typename Lat, typename T>
    LBM_HD void collide(T (&)[Lat::Q]) const {}
};

// ---------------------------------------------------------------------------
// Low-Mach state set
// ---------------------------------------------------------------------------
struct State {
    double drho;
    double ux, uy;
};

inline std::vector<State> low_mach_states() {
    const double drhos[] = {0.0, 0.01, -0.02};
    const double us[][2] = {{0.0, 0.0}, {0.05, 0.0}, {0.0, -0.03}, {0.04, 0.07}};
    std::vector<State> s;
    for (double r : drhos)
        for (const auto& u : us) s.push_back({r, u[0], u[1]});
    return s;
}

// Smallest density of low_mach_states(): 1 + drho >= 0.98. Tests that use a
// bound which depends on rho >= kRhoMin assert it.
constexpr double kRhoMin = 0.98;

// Cast a state to the compute type T and evaluate the shifted equilibrium.
template <typename T>
void inputs_from_state(const State& s, T& drho, T (&u)[2], T (&f)[L::Q]) {
    drho = static_cast<T>(s.drho);
    u[0] = static_cast<T>(s.ux);
    u[1] = static_cast<T>(s.uy);
    for (int q = 0; q < L::Q; ++q) f[q] = lbm::equilibrium_shifted<L, T>(q, drho, u);
}

// ---------------------------------------------------------------------------
// Non-equilibrium perturbation
// ---------------------------------------------------------------------------
// p_q = 1e-3 * w_q * m_q with the integer pattern m_q = ((3q + 1) mod 7) - 3,
// i.e. -2 1 -3 0 3 -1 2 -2 1 for q = 0..8. The pattern is not symmetric under
// q -> opp(q), so it changes both drho and u of the populations it is added to.
constexpr int perturbation_pattern(int q) { return ((3 * q + 1) % 7) - 3; }
static_assert(perturbation_pattern(0) == -2 && perturbation_pattern(1) == 1 &&
                  perturbation_pattern(2) == -3 && perturbation_pattern(3) == 0 &&
                  perturbation_pattern(4) == 3 && perturbation_pattern(5) == -1 &&
                  perturbation_pattern(6) == 2 && perturbation_pattern(7) == -2 &&
                  perturbation_pattern(8) == 1,
              "perturbation pattern changed");

template <typename T>
T perturbation(int q) {
    return T(1e-3) * L::template w<T>(q) * T(perturbation_pattern(q));
}

// Equilibrium populations of the state plus the perturbation.
template <typename T>
void perturbed_input(const State& s, T (&f)[L::Q]) {
    T drho, u[2];
    inputs_from_state<T>(s, drho, u, f);
    for (int q = 0; q < L::Q; ++q) f[q] = f[q] + perturbation<T>(q);
}

// ---------------------------------------------------------------------------
// Population fields on a grid
// ---------------------------------------------------------------------------
// Distinct, exactly representable values: q*1000 + node index (< 2^24).
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
// the perturbation. The moduli keep the amplitudes bounded on large grids;
// they leave grids with nx <= 9 and ny <= 5 unchanged (x % 11 = x % 9 = x,
// y % 7 = y % 5 = y there).
template <typename T>
std::vector<T> nonuniform_field(const Grid& g) {
    std::vector<T> f(static_cast<std::size_t>(L::Q * g.num_nodes()));
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x) {
            const T drho = T(0.01) * static_cast<T>(x % 11) - T(0.005) * static_cast<T>(y % 7);
            const T u[2] = {T(0.02) + T(0.004) * static_cast<T>(y % 5),
                            T(-0.01) + T(0.003) * static_cast<T>(x % 9)};
            for (int q = 0; q < L::Q; ++q)
                f[static_cast<std::size_t>(g.idx(q, x, y))] =
                    lbm::equilibrium_shifted<L, T>(q, drho, u) + perturbation<T>(q);
        }
    return f;
}

// ---------------------------------------------------------------------------
// Geometries
// ---------------------------------------------------------------------------
// Channel: walls on the full rows y = 0 and y = ny - 1.
// SingleWall: one wall node at (nx/2, ny/2) in the interior.
enum class Geometry { Channel, SingleWall };
inline const Geometry kGeometries[] = {Geometry::Channel, Geometry::SingleWall};

inline std::vector<lbm::NodeType> make_flags(const Grid& g, Geometry geo) {
    std::vector<lbm::NodeType> flags(static_cast<std::size_t>(g.num_nodes()), lbm::NodeType::Fluid);
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x) {
            const bool wall = (geo == Geometry::Channel) ? (y == 0 || y == g.ny() - 1)
                                                         : (x == g.nx() / 2 && y == g.ny() / 2);
            if (wall) flags[static_cast<std::size_t>(g.node(x, y))] = lbm::NodeType::Wall;
        }
    return flags;
}

}  // namespace lbmtest
