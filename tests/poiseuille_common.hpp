#pragma once

// Shared setup, measurement and checks of the Poiseuille validation tests
// (CPU: tests/test_poiseuille.cpp, GPU: tests/gpu/test_poiseuille_gpu.cu).
//
// ChannelSetup (as in apps/poiseuille/main.cu): nx = 4, ny = H + 2, wall rows y = 0
// and y = ny - 1 (halfway bounce-back puts the walls at y = 0.5 and
// y = ny - 1.5, channel width H), TRT<T, UniformForce<T, 2>>,
// F_x = 8 nu u_max / H^2, u_max = 0.01, rest initial state. T is the precision
// of the run (storage and compute type), double or float; the reference, the
// decay factor and the bounds are evaluated in long double for both.
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
// effective_parameters), so a wrong conversion cannot pass.
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
//    (ForceCollide.ConservesMassAndMomentumChangeIsF) for any input, here
//    with the eps of T (u = eps/2):
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
//  drho is split into its spatial mean over the fluid nodes and the fluctuation
//  around it, because the uniform density mode is neutral and is not damped
//  by r (see "Mean of drho" below):
//    |mean drho|        <= n mass_mean + meas_mean
//    |drho - mean drho| <= delta_step,rho / (1 - r) + residual_stop + ...
//  with the two bounds checked separately; u_x and u_y use the formula above.
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
//  ref_round: rounding of the long double reference, K_parabola_ref u_ld times the sum
//    of magnitudes of its terms, max u_a + F/(2 nu)(16 Lambda + 3)/12.
//
// Mean of drho (no physical assumption):
//  In exact arithmetic the total mass is conserved by (a) streaming, which is a
//  permutation of the populations between nodes, (b) halfway bounce-back, which
//  reflects a population onto its opposite direction at the same node (a
//  permutation as well), and (c) the collision, whose zeroth moment is
//  unchanged (TRT relaxes only non-conserved moments, the equilibrium has the
//  same density, and the Guo source has zero zeroth moment). Streaming and
//  bounce-back copy values and are exact also in floating point, so the
//  only mass change per step is the round-off of the collision, bounded per
//  node by tol_mass. The mass errors do not decay (the uniform mode has no
//  relaxation) but add up: after n steps
//    |mean drho| <= n mass_mean + meas_mean,
//  where mass_mean is the mean over the fluid nodes of the per-node tol_mass
//  (maximum over the checks, as for delta_step) and meas_mean the mean of the
//  rounding K_sum eps S of the measurement sum, which enters once. n is the
//  number of steps actually run. The bound is linear in n and therefore only
//  useful for runs where n eps is small compared to the tolerance wanted; it
//  is nevertheless a proven bound, which is the point. The fluctuation keeps
//  the decay assumption (b) above for the non-uniform density perturbations.
//
// No factor was chosen by trying values against data.
//
// ---------------------------------------------------------------------------
// Single precision
// ---------------------------------------------------------------------------
// The float run uses the same setup, measurement, reference and tolerance
// construction, under the same assumptions, with the eps of float in
// delta_step. Three points differ from double:
//
//  - Parameters. lbm::TRT and lbm::UniformForce compute in double and round
//    the result to T. The reference uses the stored values, and
//    effective_parameters allows one more rounding u_T per stored rate.
//
//  - Stopping rule. The threshold is the same number of eps (kStopChange). It
//    only decides when to stop: whatever its value, the distance to the fixed
//    point it leaves is bounded by residual_stop from the measured change. A
//    float run usually ends with a change of exactly zero, a fixed point of
//    the rounded scheme; its distance to the exact steady state is round-off,
//    which is what delta_step / (1 - r) bounds.
//
//  - Cases. delta_step / (1 - r) grows as eps (H + 1)^2 / nu while the slip
//    decreases as 1 / H^2, so tol / |du| grows as eps H^2 (H + 1)^2 / nu. In
//    double it is below 1e-8 in every case. In float the derived tolerance is
//    2 % to 10 % of the slip at H = 8, 30 % to 50 % at H = 16 with nu = 0.5,
//    and about the slip itself (0.9 and 1.3 times) at H = 16 with nu = 1/6.
//    A case whose tolerance is not below the slip cannot validate the slip
//    formula, so the float tests are restricted to float_cases(): H = 8 with
//    both viscosities and H = 16 with nu = 0.5. The restriction follows from
//    the derived tolerance alone, not from the measured errors, and the same
//    assertion as in double (|du| > tol_x) fails the test if a float case
//    stops resolving the slip. Comparing float against the double result
//    instead would not be sharper: the only bound derivable for that
//    difference under these assumptions is tol_float + tol_double. Wider
//    channels and smaller viscosities are therefore not validated in float by
//    this construction.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "lbm/collision/trt.hpp"
#include "lbm/equilibrium.hpp"
#include "lbm/force.hpp"
#include "lbm/grid.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "lbm/moments.hpp"
#include "lbm/propagation/ab_pull.hpp"
#include "support/error_bounds.hpp"
#include "support/fields.hpp"

namespace poiseuille {

using namespace lbmtest;  // L, Grid, ld, IdentityCollision, the K_* constants, ...

template <typename T>
using Force = lbm::UniformForce<T, L::D>;
template <typename T>
using Collision = lbm::TRT<T, Force<T>>;

constexpr double kUmax = 0.01;
constexpr int kNx = 4;
constexpr int kCheckEvery = 1000;
constexpr long kStepCap = 1000000;
// Stopping threshold on the change of u_x over one check window: 1e-14 u_max in
// double (about 45 eps u_max) and the same number of eps in T (the ratio of
// the epsilons is a power of two, so the double value is unchanged).
template <typename T>
constexpr double kStopChange =
    1e-14 * kUmax *
    (static_cast<double>(std::numeric_limits<T>::epsilon()) / std::numeric_limits<double>::epsilon());
static_assert(kStopChange<double> == 1e-14 * kUmax, "the double stopping rule must not change");

// Operation-count factors K_chain, K_src, K_w, K_sum, K_e, K_div are shared and
// derived in support/error_bounds.hpp. This one belongs to this file:
// the long double reference chain (the parabola, the slip and the effective
// parameters) has at most 12 operations, in units of u_ld.
constexpr ld K_parabola_ref = 16.0L;
static_assert(K_parabola_ref <= 16.0L, "every factor must not exceed 16");

// Rounding of the conversion double -> T of a stored parameter (exact for double).
template <typename T>
constexpr ld kUcast = std::is_same_v<T, double> ? 0.0L : kU<T>;
constexpr ld kPi = 3.14159265358979323846264338327950288L;

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

template <typename T>
struct ChannelSetup {
    Case c;
    Grid g;
    std::vector<lbm::NodeType> flags;
    Force<T> force;
    Collision<T> coll;

    explicit ChannelSetup(const Case& cs)
        : c(cs),
          g(kNx, cs.H + 2),
          flags(make_flags(g, Geometry::Channel)),
          force(make_force(cs)),
          coll(Collision<T>::template from_viscosity<L>(cs.nu, cs.lambda, force)) {}

    static Force<T> make_force(const Case& cs) {
        const double fc[2] = {8.0 * cs.nu * kUmax / (static_cast<double>(cs.H) * cs.H), 0.0};
        return Force<T>::create(fc);
    }
};

template <typename T>
struct Fields {
    std::vector<T> ux, uy, drho;  // index y * nx + x (walls: 0)
};

struct Bounds {
    ld ux = 0, uy = 0, rho = 0;  // delta_step per field
    ld mass_mean = 0;            // mean over the fluid nodes of tol_mass (max over checks)
    ld meas_mean = 0;            // mean over the fluid nodes of the measurement rounding of drho
};

// Per-node contributions to the mean bounds, summed by measure().
struct NodeMass {
    ld mass = 0, meas = 0;
};

// delta_step of one node, from its gathered (collision input) populations.
template <typename T>
NodeMass accumulate_bounds(const T (&f)[L::Q], T drho, const T (&u)[L::D], const T (&F)[L::D],
                           ld wp, ld wm, Bounds& b) {
    const ld eps = kEps<T>;
    const ld uT = kU<T>;
    ld N[L::Q];
    ld S = 0;
    for (int q = 0; q < L::Q; ++q) {
        N[q] = std::abs(static_cast<ld>(f[q])) +
               std::abs(static_cast<ld>(lbm::equilibrium_shifted<L, T>(q, drho, u)));
        S += std::abs(static_cast<ld>(f[q]));
    }
    ld sum_scale = 0, sum_chain = 0, sum_W = 0;
    ld sc[L::D] = {0, 0}, schain[L::D] = {0, 0}, sW[L::D] = {0, 0};
    for (int q = 0; q < L::Q; ++q) {
        const ld scale = scale_e(q, drho, u);
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
    const ld tol_mass = wp * (K_sum * eps * S + (K_e * eps + K_w * uT) * sum_scale) +
                        K_chain * uT * sum_chain + (K_src + K_w) * uT * sum_W;
    const ld d_rho = tol_mass + K_sum * eps * S;  // + measurement
    const ld rho = 1.0L + static_cast<ld>(drho);
    ld du[L::D];
    for (int d = 0; d < L::D; ++d) {
        const ld Fd = std::abs(static_cast<ld>(F[d]));
        const ld meas_j = K_sum * eps * S + K_div * uT * (S + 0.5L * Fd);
        const ld tol_j = wm * (K_sum * eps * S + (K_e * eps + K_w * uT) * sc[d] +
                               K_div * uT * (S + 0.5L * Fd)) +
                         K_chain * uT * schain[d] + (K_src + K_w) * uT * sW[d];
        du[d] = (tol_j + meas_j + std::abs(static_cast<ld>(u[d])) * d_rho) / rho;
    }
    b.ux = std::max(b.ux, du[0]);
    b.uy = std::max(b.uy, du[1]);
    b.rho = std::max(b.rho, d_rho);
    return {tol_mass, K_sum * eps * S};
}

// Fields of the stored post-collision populations, measured on the gathered
// populations (see the file comment); also accumulates the step bounds.
template <typename T>
Fields<T> measure(const ChannelSetup<T>& s, const std::vector<T>& stored, Bounds& bounds) {
    const Grid& g = s.g;
    std::vector<T> gathered = stored;
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x)
            lbm::pull_stream_collide<L, T>(g, s.flags.data(), stored.data(), gathered.data(),
                                           IdentityCollision{}, x, y);
    Fields<T> r;
    const std::size_t n = static_cast<std::size_t>(g.num_nodes());
    r.ux.assign(n, T(0));
    r.uy.assign(n, T(0));
    r.drho.assign(n, T(0));
    const T F[2] = {s.force.F(0), s.force.F(1)};
    ld sum_mass = 0, sum_meas = 0;
    long fluid = 0;
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x) {
            if (s.flags[static_cast<std::size_t>(g.node(x, y))] == lbm::NodeType::Wall) continue;
            T f[L::Q], u[2], drho;
            for (int q = 0; q < L::Q; ++q) f[q] = gathered[static_cast<std::size_t>(g.idx(q, x, y))];
            lbm::moments_shifted_forced<L, T>(f, F, drho, u);
            const std::size_t i = static_cast<std::size_t>(g.node(x, y));
            r.ux[i] = u[0];
            r.uy[i] = u[1];
            r.drho[i] = drho;
            const NodeMass m = accumulate_bounds(f, drho, u, F, s.coll.omega_plus(),
                                                 s.coll.omega_minus(), bounds);
            sum_mass += m.mass;
            sum_meas += m.meas;
            ++fluid;
        }
    bounds.mass_mean = std::max(bounds.mass_mean, sum_mass / static_cast<ld>(fluid));
    bounds.meas_mean = std::max(bounds.meas_mean, sum_meas / static_cast<ld>(fluid));
    return r;
}

// Effective parameters of the stored rates, and their agreement with the
// nominal ones within the derived rounding bounds. With u = eps/2 of double
// and the conversion of lbm::TRT, computed in double (tau+ = fl(fl(3 nu) + 1/2),
// a = fl(tau+ - 1/2), tau- = fl(fl(lambda/a) + 1/2), omega = fl(1/tau)):
//   tau+_eff = 1/omega+ has relative error <= 3u (3 nu: 1, + 1/2: 1, 1/tau: 1),
//   so (tau+ - 1/2) has relative error <= 3u tau+/(3 nu) = 3u (1 + 1/(6 nu)).
//   a has relative error <= 3u tau+/(3 nu) (2u tau+ from tau+, u a from the
//   subtraction), q = lambda/a adds u: q has relative error
//   <= u (3 tau+/(3 nu) + 1); tau-_eff - 1/2 adds 2u tau-/q (the sum and 1/tau
//   roundings, each u tau-).
// Lambda_eff = (tau+ - 1/2)(tau- - 1/2) has the sum of the relative errors
// of its two factors (first order; second order is covered by K_parabola_ref u_ld).
// For T = float each omega is then rounded to T, one more relative error
// u_c = eps_T/2 of tau_eff (u_c = 0 for double, where the cast is exact):
// 3u becomes 3u + u_c in the tau+ terms and 2u becomes 2u + u_c in the tau-
// term. The a and q of the conversion never see the stored float rates, so
// rel_q is unchanged.
struct Effective {
    ld nu, lambda, tau_plus;
};

template <typename T>
Effective effective_parameters(const ChannelSetup<T>& s) {
    const ld uD = kU<double>;   // the conversion is computed in double
    const ld uC = kUcast<T>;    // and each rate is then stored as T
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
    const ld rel_nu = (3.0L * uD + uC) * (1.0L + 1.0L / (6.0L * nu));
    const ld rel_q = uD * (3.0L * tp_nom / (3.0L * nu) + 1.0L);
    const ld rel_lambda = rel_nu + rel_q + (2.0L * uD + uC) * tm_nom / q_nom;
    EXPECT_LE(std::abs(e.nu / nu - 1.0L), rel_nu + K_parabola_ref * kULd) << "effective nu";
    EXPECT_LE(std::abs(e.lambda / lam - 1.0L), rel_lambda + K_parabola_ref * kULd) << "effective lambda";
    return e;
}

// Stepper interface used by check_case:
//   void advance(int n)                          n steps with alternating buffers
//   const std::vector<T>& state()                stored populations (host copy)
template <typename T, typename Stepper>
void check_case(const ChannelSetup<T>& s, Stepper& stepper) {
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
    Fields<T> prev = measure(s, stepper.state(), bounds);  // rest state
    Fields<T> cur = prev;
    T change_x = 0, change_y = 0, change_rho = 0;
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
        if (static_cast<double>(change_x) <= kStopChange<T>) {
            converged = true;
            break;
        }
    }
    ASSERT_TRUE(converged) << "step cap " << kStepCap << " reached, change = " << change_x;

    ld ua_max = 0;
    for (int y = 1; y < ny - 1; ++y) ua_max = std::max(ua_max, analytic(y));
    const ld ref_round = K_parabola_ref * kULd * (ua_max + Fx / (2.0L * eff.nu) * (16.0L * eff.lambda + 3.0L) / 12.0L);

    auto tail = [&](T change) { return static_cast<ld>(change) * rK / (1.0L - rK); };
    const ld tol_x = bounds.ux / (1.0L - r) + tail(change_x) + ref_round;
    const ld tol_y = bounds.uy / (1.0L - r) + tail(change_y);
    // Mean (accumulates over the steps run) and fluctuation (decays), see "Mean of drho".
    const ld tol_rho_mean = static_cast<ld>(steps) * bounds.mass_mean + bounds.meas_mean;
    const ld tol_rho = bounds.rho / (1.0L - r) + tail(change_rho);  // fluctuation

    ld err_x = 0, err_y = 0, err_rho = 0, mean_rho = 0;
    for (int y = 1; y < ny - 1; ++y)
        for (int x = 0; x < kNx; ++x) {
            const std::size_t i = static_cast<std::size_t>(g.node(x, y));
            err_x = std::max(err_x, std::abs(static_cast<ld>(cur.ux[i]) - (analytic(y) + du)));
            err_y = std::max(err_y, std::abs(static_cast<ld>(cur.uy[i])));
            mean_rho += static_cast<ld>(cur.drho[i]);
        }
    mean_rho /= static_cast<ld>((ny - 2) * kNx);
    for (int y = 1; y < ny - 1; ++y)
        for (int x = 0; x < kNx; ++x) {
            const std::size_t i = static_cast<std::size_t>(g.node(x, y));
            err_rho = std::max(err_rho, std::abs(static_cast<ld>(cur.drho[i]) - mean_rho));
        }

    std::cout << "[ info ] " << case_name(c) << (sizeof(T) == sizeof(float) ? " float" : " double")
              << ": steps " << steps << ", du/umax "
              << static_cast<double>(du / kUmax) << ", err_x/umax "
              << static_cast<double>(err_x / kUmax) << " (tol "
              << static_cast<double>(tol_x / kUmax) << "), err_y " << static_cast<double>(err_y)
              << " (tol " << static_cast<double>(tol_y) << "), err_rho "
              << static_cast<double>(err_rho) << " (tol " << static_cast<double>(tol_rho)
              << "), mean_rho " << static_cast<double>(std::abs(mean_rho)) << " (tol "
              << static_cast<double>(tol_rho_mean) << ")\n";

    EXPECT_LE(err_x, tol_x) << "u_x vs u_a + du, " << case_name(c);
    EXPECT_LE(err_y, tol_y) << "u_y, " << case_name(c);
    EXPECT_LE(err_rho, tol_rho) << "drho fluctuation, " << case_name(c);
    EXPECT_LE(std::abs(mean_rho), tol_rho_mean) << "drho mean, " << case_name(c);

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

// Cases of the float tests: those of all_cases() whose derived float tolerance
// is below the slip (see "Single precision" in the file comment).
inline std::vector<Case> float_cases() {
    std::vector<Case> v;
    for (const Case& c : all_cases())
        if (c.H == 8 || c.nu == 0.5) v.push_back(c);
    return v;
}

}  // namespace poiseuille
