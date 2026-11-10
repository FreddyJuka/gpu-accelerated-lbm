#pragma once

// Shared setup, measurement and checks of the Poiseuille validation tests
// (CPU: tests/test_poiseuille.cpp, GPU: tests/gpu/test_poiseuille_gpu.cu).
//
// ChannelSetup (as in apps/poiseuille/main.cu): nx = 4, ny = H + 2, wall rows y = 0
// and y = ny - 1 (halfway bounce-back puts the walls at y = 0.5 and
// y = ny - 1.5, channel width H), TRT<double, UniformForce<double, 2>>,
// F_x = 8 nu u_max / H^2, u_max = 0.01, rest initial state.
//
// ---------------------------------------------------------------------------
// Reference solution
// ---------------------------------------------------------------------------
// The steady TRT solution with halfway bounce-back is the parabola
//   u_a(y) = F_x / (2 nu) (y - 1/2)(ny - 3/2 - y)
// shifted by the uniform slip
//   du = F_x / (2 nu) (16 Lambda - 3) / 12
// (Ginzburg, Verhaeghe, d'Humieres, Commun. Comput. Phys. 3(2) (2008),
// Eq. (2.29) with Table 1, bounce-back delta_q = 1/2; measured to round-off
// with apps/poiseuille for Lambda = 3/16 and 1/4). nu and Lambda are the
// values effectively stored in the collision object: nu_eff = (1/omega_plus -
// 1/2)/3 and Lambda_eff = (1/omega_plus - 1/2)(1/omega_minus - 1/2), evaluated
// in long double. This removes the rounding of the conversion nu, lambda ->
// rates from the comparison; that conversion is tested in
// test_collision_trt.cpp, and here the effective values are asserted to agree
// with the nominal ones within their derived rounding bounds (see
// check_effective_parameters), so a wrong conversion cannot pass.
//
// ---------------------------------------------------------------------------
// Velocity measurement
// ---------------------------------------------------------------------------
// The stored populations are post-collision. The Guo half-force velocity
// u = (j + F/2)/rho is defined on the collision input, i.e. after the gather
// (streaming and bounce-back). The collision adds F to the momentum, so
// moments_shifted_forced applied directly to the stored populations would be
// off by about F. The velocity is therefore measured by running the gather of
// pull_stream_collide (identity collision) on the stored populations and
// applying moments_shifted_forced to the result.
//
// ---------------------------------------------------------------------------
// Tolerance
// ---------------------------------------------------------------------------
//   tol = delta_step / (1 - r) + residual_stop + ref_round
// for each of u_x (against u_a + du), u_y and drho (against 0), where
//
//  delta_step: bound on the round-off one step (plus the measurement itself)
//    introduces into the measured field, from operation counts. The momentum
//    added by the collision deviates from F by at most tol_j, and the mass
//    by at most tol_mass; these are the bounds derived in test_force.cpp
//    (ForceCollide.ConservesMassAndMomentumChangeIsF) for any input, here in
//    double (u = eps/2):
//      tol_mass = w+ (K_sum eps S + (K_e eps + K_w u) sum_q scale_q)
//                 + K_chain u sum_q chain_q + (K_src + K_w) u sum_q W_q
//      tol_j,d  = w- (K_sum eps S + (K_e eps + K_w u) sum_q |c_qd| scale_q
//                     + K_div u (S + |F_d|/2))
//                 + K_chain u sum_q |c_qd| chain_q + (K_src + K_w) u sum_q |c_qd| W_q
//    with S = sum|f|, N_q = |f_q| + |feq_q|, scale_q the sum of magnitudes of
//    the equilibrium terms, W_q the sum of magnitudes of the source terms,
//    chain_q = M_q + N_q + N_opp, M_q = |f_q| + (w+ + w-)/2 (N_q + N_opp) + W_q
//    (all partial results of the update of out_q). The measurement adds
//    K_sum eps S + K_div u (S + |F_d|/2) to j_d and K_sum eps S to drho, and
//    u = (j + F/2)/rho gives delta_u,d = (tol_j,d + meas_j,d +
//    |u_d| (tol_mass + meas_rho))/rho. The bound is evaluated on the gathered
//    populations of every fluid node at every check and the maximum over
//    nodes and checks is used. (The velocity approaches its steady value
//    monotonically from rest, so the sampled magnitudes bound those between
//    checks; this is part of the assumption below.)
//
//  r: decay factor per step of the slowest diffusive mode of the channel,
//    r = exp(-nu_eff pi^2 / (H + 1)^2). Two deliberate differences from the
//    simplification 1 - nu pi^2 / H^2: exp(-x) >= 1 - x, so r is not smaller
//    than the linearised value, and the width H + 1 (distance between the
//    wall nodes) is not smaller than the channel width of any tested case
//    (H_eff^2 = H^2 - 1 + 16 Lambda/3 <= H^2 + 1/3 for Lambda <= 1/4), so
//    the estimated decay is never faster than the physical one.
//    THIS IS A PHYSICAL ESTIMATE of the slowest hydrodynamic mode, NOT a
//    proven contraction bound for the full (non-normal) LBM operator. The
//    test relies on it, and on two further assumptions: (a) round-off enters
//    the measured fields through the conserved moments with the bounds above
//    and perturbations of the non-conserved moments are of the same size and
//    relax at least as fast; (b) the same r applies to the u_y and drho
//    perturbations (acoustic modes decay at least as fast as the shear mode).
//    The steady state of the exact-arithmetic scheme is taken to be exactly
//    u_a + du, u_y = 0, drho = 0 (the published result stated above).
//
//  residual_stop: distance to the fixed point left by the stopping rule. The
//    last check window of K steps changed the field by `change`, so with the
//    error decaying by r per step the remaining error is
//    change r^K/(1 - r^K). `change` is the measured final change of the
//    respective field.
//
//  ref_round: rounding of the long double reference, K_ref u_ld times the sum
//    of magnitudes of its terms, max u_a + F/(2 nu)(16 Lambda + 3)/12.
//
// No factor was chosen by trying values against data.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "lbm/collision/trt.hpp"
#include "lbm/equilibrium.hpp"
#include "lbm/force.hpp"
#include "lbm/grid.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "lbm/moments.hpp"
#include "lbm/propagation/ab_pull.hpp"

namespace poiseuille {

using L = lbm::D2Q9;
using Grid = lbm::Grid2D<L>;
using Force = lbm::UniformForce<double, L::D>;
using Collision = lbm::TRT<double, Force>;
using ld = long double;

static_assert(std::numeric_limits<ld>::digits >= 64,
              "long double must be extended precision for the reference");

constexpr double kUmax = 0.01;
constexpr int kNx = 4;
constexpr int kCheckEvery = 1000;
constexpr long kStepCap = 1000000;
constexpr double kStopChange = 1e-14 * kUmax;

// Operation-count factors (see the comment above and test_force.cpp).
constexpr ld K_chain = 14.0L;  // roundings in the update of one out_q
constexpr ld K_src = 10.0L;    // roundings in guo_source_parts, D = 2
constexpr ld K_w = 1.0L;       // representation of the weights
constexpr ld K_sum = L::Q / 2.0L;  // sum of Q terms
constexpr ld K_e = 6.0L;       // shifted equilibrium
constexpr ld K_div = 3.0L;     // u = (j + F/2) * inv_rho
constexpr ld K_ref = 16.0L;    // long double reference chain (at most 12 operations)
static_assert(K_chain <= 16.0L && K_src <= 16.0L && K_w <= 16.0L && K_sum <= 16.0L &&
                  K_e <= 16.0L && K_div <= 16.0L && K_ref <= 16.0L,
              "every factor must not exceed 16");

constexpr ld kEps = std::numeric_limits<double>::epsilon();
constexpr ld kU = 0.5L * kEps;
constexpr ld kUld = 0.5L * std::numeric_limits<ld>::epsilon();
constexpr ld kPi = 3.14159265358979323846264338327950288L;

struct IdentityCollision {
    template <typename Lat, typename T>
    LBM_HD void collide(T (&)[Lat::Q]) const {}
};

struct Case {
    int H;
    double nu;
    double lambda;
};

inline std::string case_name(const Case& c);

// Readable parameter output for gtest (found by argument-dependent lookup).
inline std::ostream& operator<<(std::ostream& os, const Case& c) {
    return os << case_name(c);
}

inline std::string case_name(const Case& c) {
    auto tag = [](double v) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.4g", v);
        std::string s(buf);
        for (char& ch : s)
            if (ch == '.' || ch == '-') ch = '_';
        return s;
    };
    return "H" + std::to_string(c.H) + "_nu" + tag(c.nu) + "_lambda" + tag(c.lambda);
}

struct ChannelSetup {
    Case c;
    Grid g;
    std::vector<lbm::NodeType> flags;
    Force force;
    Collision coll;

    explicit ChannelSetup(const Case& cs)
        : c(cs),
          g(kNx, cs.H + 2),
          flags(make_flags(kNx, cs.H + 2)),
          force(make_force(cs)),
          coll(Collision::from_viscosity<L>(cs.nu, cs.lambda, force)) {}

    static std::vector<lbm::NodeType> make_flags(int nx, int ny) {
        std::vector<lbm::NodeType> f(static_cast<std::size_t>(nx * ny), lbm::NodeType::Fluid);
        for (int x = 0; x < nx; ++x) {
            f[static_cast<std::size_t>(x)] = lbm::NodeType::Wall;
            f[static_cast<std::size_t>((ny - 1) * nx + x)] = lbm::NodeType::Wall;
        }
        return f;
    }
    static Force make_force(const Case& cs) {
        const double fc[2] = {8.0 * cs.nu * kUmax / (static_cast<double>(cs.H) * cs.H), 0.0};
        return Force::create(fc);
    }
};

struct Fields {
    std::vector<double> ux, uy, drho;  // index y * nx + x (walls: 0)
};

struct Bounds {
    ld ux = 0, uy = 0, rho = 0;  // delta_step per field
};

// Sum of magnitudes of the terms of s_anti + s_sym (test_force.cpp).
inline ld source_scale(int q, const double (&u)[L::D], const double (&F)[L::D]) {
    const ld a = L::inv_cs2<ld>();
    ld cu = 0, cF = 0, uF = 0;
    for (int d = 0; d < L::D; ++d) {
        cu += std::abs(static_cast<ld>(L::c(q, d))) * std::abs(static_cast<ld>(u[d]));
        cF += std::abs(static_cast<ld>(L::c(q, d))) * std::abs(static_cast<ld>(F[d]));
        uF += std::abs(static_cast<ld>(u[d])) * std::abs(static_cast<ld>(F[d]));
    }
    return L::w<ld>(q) * (a * a * cu * cF + a * uF + a * cF);
}

// Sum of magnitudes of the terms of the shifted equilibrium (test_equilibrium.cpp).
inline ld equilibrium_scale(int q, double drho_d, const double (&u_d)[L::D]) {
    const ld drho = drho_d;
    const ld rho = 1.0L + drho;
    const ld a = L::inv_cs2<ld>();
    ld cu = 0, uu = 0;
    for (int d = 0; d < L::D; ++d) {
        cu += L::c(q, d) * static_cast<ld>(u_d[d]);
        uu += static_cast<ld>(u_d[d]) * static_cast<ld>(u_d[d]);
    }
    return L::w<ld>(q) *
           (std::abs(drho) + rho * (a * std::abs(cu) + 0.5L * a * a * cu * cu + 0.5L * a * uu));
}

// delta_step of one node, from its gathered (collision input) populations.
inline void accumulate_bounds(const double (&f)[L::Q], double drho, const double (&u)[L::D],
                              const double (&F)[L::D], ld wp, ld wm, Bounds& b) {
    ld N[L::Q];
    ld S = 0;
    for (int q = 0; q < L::Q; ++q) {
        N[q] = std::abs(static_cast<ld>(f[q])) +
               std::abs(static_cast<ld>(lbm::equilibrium_shifted<L, double>(q, drho, u)));
        S += std::abs(static_cast<ld>(f[q]));
    }
    ld sum_scale = 0, sum_chain = 0, sum_W = 0;
    ld sc[L::D] = {0, 0}, schain[L::D] = {0, 0}, sW[L::D] = {0, 0};
    for (int q = 0; q < L::Q; ++q) {
        const ld scale = equilibrium_scale(q, drho, u);
        const ld W = source_scale(q, u, F);
        const ld Np = N[q] + N[L::opp(q)];
        const ld M = std::abs(static_cast<ld>(f[q])) + 0.5L * (wp + wm) * Np + W;
        const ld chain = M + Np;
        sum_scale += scale;
        sum_chain += chain;
        sum_W += W;
        for (int d = 0; d < L::D; ++d) {
            const ld ac = std::abs(static_cast<ld>(L::c(q, d)));
            sc[d] += ac * scale;
            schain[d] += ac * chain;
            sW[d] += ac * W;
        }
    }
    const ld tol_mass = wp * (K_sum * kEps * S + (K_e * kEps + K_w * kU) * sum_scale) +
                        K_chain * kU * sum_chain + (K_src + K_w) * kU * sum_W;
    const ld d_rho = tol_mass + K_sum * kEps * S;  // + measurement
    const ld rho = 1.0L + static_cast<ld>(drho);
    ld du[L::D];
    for (int d = 0; d < L::D; ++d) {
        const ld Fd = std::abs(static_cast<ld>(F[d]));
        const ld meas_j = K_sum * kEps * S + K_div * kU * (S + 0.5L * Fd);
        const ld tol_j = wm * (K_sum * kEps * S + (K_e * kEps + K_w * kU) * sc[d] +
                               K_div * kU * (S + 0.5L * Fd)) +
                         K_chain * kU * schain[d] + (K_src + K_w) * kU * sW[d];
        du[d] = (tol_j + meas_j + std::abs(static_cast<ld>(u[d])) * d_rho) / rho;
    }
    b.ux = std::max(b.ux, du[0]);
    b.uy = std::max(b.uy, du[1]);
    b.rho = std::max(b.rho, d_rho);
}

// Fields of the stored post-collision populations, measured on the gathered
// populations (see the file comment); also accumulates the step bounds.
inline Fields measure(const ChannelSetup& s, const std::vector<double>& stored, Bounds& bounds) {
    const Grid& g = s.g;
    std::vector<double> gathered = stored;
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x)
            lbm::pull_stream_collide<L, double>(g, s.flags.data(), stored.data(), gathered.data(),
                                                IdentityCollision{}, x, y);
    Fields r;
    const std::size_t n = static_cast<std::size_t>(g.num_nodes());
    r.ux.assign(n, 0.0);
    r.uy.assign(n, 0.0);
    r.drho.assign(n, 0.0);
    const double F[2] = {s.force.F(0), s.force.F(1)};
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x) {
            if (s.flags[static_cast<std::size_t>(g.node(x, y))] == lbm::NodeType::Wall) continue;
            double f[L::Q], u[2], drho;
            for (int q = 0; q < L::Q; ++q) f[q] = gathered[static_cast<std::size_t>(g.idx(q, x, y))];
            lbm::moments_shifted_forced<L, double>(f, F, drho, u);
            const std::size_t i = static_cast<std::size_t>(g.node(x, y));
            r.ux[i] = u[0];
            r.uy[i] = u[1];
            r.drho[i] = drho;
            accumulate_bounds(f, drho, u, F, s.coll.omega_plus(), s.coll.omega_minus(), bounds);
        }
    return r;
}

// Effective parameters of the stored rates, and their agreement with the
// nominal ones within the derived rounding bounds. With u = eps/2 and the
// conversion of lbm::TRT (tau+ = fl(fl(3 nu) + 1/2), a = fl(tau+ - 1/2),
// tau- = fl(fl(lambda/a) + 1/2), omega = fl(1/tau)):
//   tau+_eff = 1/omega+ has relative error <= 3u (3 nu: 1, + 1/2: 1, 1/tau: 1),
//   so (tau+ - 1/2) has relative error <= 3u tau+/(3 nu) = 3u (1 + 1/(6 nu)).
//   a has relative error <= 3u tau+/(3 nu) (2u tau+ from tau+, u a from the
//   subtraction), q = lambda/a adds u: q has relative error
//   <= u (3 tau+/(3 nu) + 1); tau-_eff - 1/2 adds 2u tau-/q (the sum and 1/tau
//   roundings, each u tau-).
// Lambda_eff = (tau+ - 1/2)(tau- - 1/2) has the sum of the relative errors
// of its two factors (first order; second order is covered by K_ref u_ld).
struct Effective {
    ld nu, lambda, tau_plus;
};

inline Effective effective_parameters(const ChannelSetup& s) {
    const ld tp = 1.0L / static_cast<ld>(s.coll.omega_plus());
    const ld tm = 1.0L / static_cast<ld>(s.coll.omega_minus());
    Effective e;
    e.tau_plus = tp;
    e.nu = (tp - 0.5L) / 3.0L;
    e.lambda = (tp - 0.5L) * (tm - 0.5L);

    const ld nu = s.c.nu, lam = s.c.lambda;
    const ld tp_nom = 3.0L * nu + 0.5L;
    const ld q_nom = lam / (3.0L * nu);
    const ld tm_nom = q_nom + 0.5L;
    const ld rel_nu = 3.0L * kU * (1.0L + 1.0L / (6.0L * nu));
    const ld rel_q = kU * (3.0L * tp_nom / (3.0L * nu) + 1.0L);
    const ld rel_lambda = rel_nu + rel_q + 2.0L * kU * tm_nom / q_nom;
    EXPECT_LE(std::abs(e.nu / nu - 1.0L), rel_nu + K_ref * kUld) << "effective nu";
    EXPECT_LE(std::abs(e.lambda / lam - 1.0L), rel_lambda + K_ref * kUld) << "effective lambda";
    return e;
}

// Stepper interface used by check_case:
//   void advance(int n)                          n steps with alternating buffers
//   const std::vector<double>& state()           stored populations (host copy)
template <typename Stepper>
void check_case(const ChannelSetup& s, Stepper& stepper) {
    const Case& c = s.c;
    const Grid& g = s.g;
    const int ny = g.ny();
    const Effective eff = effective_parameters(s);
    const ld Fx = s.force.F(0);

    const ld width = static_cast<ld>(c.H) + 1.0L;
    const ld r = std::exp(-eff.nu * kPi * kPi / (width * width));
    const ld rK = std::pow(r, static_cast<ld>(kCheckEvery));
    ASSERT_LT(r, 1.0L);

    auto analytic = [&](int y) {
        return Fx / (2.0L * eff.nu) * (static_cast<ld>(y) - 0.5L) * (ny - 1.5L - static_cast<ld>(y));
    };
    const ld du = Fx / (2.0L * eff.nu) * (16.0L * eff.lambda - 3.0L) / 12.0L;

    Bounds bounds;
    Fields prev = measure(s, stepper.state(), bounds);  // rest state
    Fields cur = prev;
    double change_x = 0, change_y = 0, change_rho = 0;
    long steps = 0;
    bool converged = false;
    while (steps < kStepCap) {
        stepper.advance(kCheckEvery);
        steps += kCheckEvery;
        cur = measure(s, stepper.state(), bounds);
        change_x = change_y = change_rho = 0;
        for (std::size_t i = 0; i < cur.ux.size(); ++i) {
            if (s.flags[i] == lbm::NodeType::Wall) continue;
            change_x = std::max(change_x, std::abs(cur.ux[i] - prev.ux[i]));
            change_y = std::max(change_y, std::abs(cur.uy[i] - prev.uy[i]));
            change_rho = std::max(change_rho, std::abs(cur.drho[i] - prev.drho[i]));
        }
        prev = cur;
        if (change_x <= kStopChange) {
            converged = true;
            break;
        }
    }
    ASSERT_TRUE(converged) << "step cap " << kStepCap << " reached, change = " << change_x;

    ld ua_max = 0;
    for (int y = 1; y < ny - 1; ++y) ua_max = std::max(ua_max, analytic(y));
    const ld ref_round = K_ref * kUld * (ua_max + Fx / (2.0L * eff.nu) * (16.0L * eff.lambda + 3.0L) / 12.0L);

    auto tail = [&](double change) { return static_cast<ld>(change) * rK / (1.0L - rK); };
    const ld tol_x = bounds.ux / (1.0L - r) + tail(change_x) + ref_round;
    const ld tol_y = bounds.uy / (1.0L - r) + tail(change_y);
    const ld tol_rho = bounds.rho / (1.0L - r) + tail(change_rho);

    ld err_x = 0, err_y = 0, err_rho = 0;
    for (int y = 1; y < ny - 1; ++y)
        for (int x = 0; x < kNx; ++x) {
            const std::size_t i = static_cast<std::size_t>(g.node(x, y));
            err_x = std::max(err_x, std::abs(static_cast<ld>(cur.ux[i]) - (analytic(y) + du)));
            err_y = std::max(err_y, std::abs(static_cast<ld>(cur.uy[i])));
            err_rho = std::max(err_rho, std::abs(static_cast<ld>(cur.drho[i])));
        }

    std::cout << "[ info ] " << case_name(c) << ": steps " << steps << ", du/umax "
              << static_cast<double>(du / kUmax) << ", err_x/umax "
              << static_cast<double>(err_x / kUmax) << " (tol "
              << static_cast<double>(tol_x / kUmax) << "), err_y " << static_cast<double>(err_y)
              << " (tol " << static_cast<double>(tol_y) << "), err_rho "
              << static_cast<double>(err_rho) << " (tol " << static_cast<double>(tol_rho) << ")\n";

    EXPECT_LE(err_x, tol_x) << "u_x vs u_a + du, " << case_name(c);
    EXPECT_LE(err_y, tol_y) << "u_y, " << case_name(c);
    EXPECT_LE(err_rho, tol_rho) << "drho, " << case_name(c);

    // The tolerance must be below the slip, otherwise a result that ignores
    // the slip could pass.
    if (c.lambda != 3.0 / 16.0) {
        EXPECT_GT(std::abs(du), tol_x) << "slip not resolved, " << case_name(c);
    }
}

inline std::vector<Case> all_cases() {
    std::vector<Case> v;
    for (int H : {8, 16})
        for (double nu : {1.0 / 6.0, 0.5})
            for (double lambda : {3.0 / 16.0, 1.0 / 4.0, 1.0 / 12.0}) v.push_back({H, nu, lambda});
    return v;
}

}  // namespace poiseuille
