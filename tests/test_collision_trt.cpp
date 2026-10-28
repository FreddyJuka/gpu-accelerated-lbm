#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "lbm/collision/trt.hpp"
#include "lbm/equilibrium.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "lbm/moments.hpp"

using L = lbm::D2Q9;
using ld = long double;

namespace {

struct State {
    double drho;
    double ux, uy;
};

// Same low-Mach state set as test_moments.cpp.
std::vector<State> low_mach_states() {
    const double drhos[] = {0.0, 0.01, -0.02};
    const double us[][2] = {{0.0, 0.0}, {0.05, 0.0}, {0.0, -0.03}, {0.04, 0.07}};
    std::vector<State> s;
    for (double r : drhos)
        for (const auto& u : us) s.push_back({r, u[0], u[1]});
    return s;
}

struct Params {
    double nu;
    double lambda;
};
constexpr Params P1{0.01, 3.0 / 16.0};  // tau+ = 0.53, tau- = 6.75: omega+ != omega-
constexpr Params P2{1.0 / 6.0, 0.25};   // tau+ = tau- = 1: omega+ = omega- = 1

static_assert(std::numeric_limits<ld>::digits >= 64,
              "long double must be extended precision for the reference");

// ---------------------------------------------------------------------------
// Error-bound constants. u = eps/2 is the unit roundoff of the type in
// question. Each is an upper bound on a derived count, with the headroom
// stated; none is fitted to data.
// ---------------------------------------------------------------------------
// Summation of Q terms (moments_shifted, test_moments.cpp): (Q-1) u = (Q-1)/2
// eps, plus 1/2 eps for the O(n u) second-order factor.
constexpr ld K_sum = L::Q / 2.0L;
// u_d = j_d * inv_rho (test_moments.cpp): 1.5 eps |u| plus 1/2 eps headroom.
constexpr ld K_u = 2.0L;
// Shifted equilibrium (test_equilibrium.cpp): 10 u = 5 eps on the sum of term
// magnitudes, plus 1 eps for second-order factors.
constexpr ld K_e = 6.0L;
// Second-order factor on the rounding terms of the collision chain: the chain
// has at most 8 roundings per quantity, (1+u)^8 - 1 - 8u < 40 u^2, and the
// bounds use first-order magnitudes. Doubling the u-terms covers this by a
// factor ~1/(40 u) in float and more in double.
constexpr ld K_h = 2.0L;
// Parameter relations (see the derivations at expected_*):
//   omega_plus  from nu : 3 u + u_T = 1.5 eps + 0.5 eps_T, bounded by 2 eps + 0.5 eps_T
//   omega_minus         : (3 u tau+/d + 3 u) + u_T, bounded by 2 eps (tau+/d + 1) + 0.5 eps_T
//   reference in ld     : at most 3 u_ld = 1.5 eps_ld, bounded by 2 eps_ld
constexpr ld K_par = 2.0L;
static_assert(K_sum <= 16.0L && K_u <= 16.0L && K_e <= 16.0L && K_h <= 16.0L &&
                  K_par <= 16.0L && K_sum + 1.0L <= 16.0L,
              "every factor must not exceed 16");

// TRT can only be obtained through its named constructors.
template <typename T>
constexpr bool trt_is_sealed =
    !std::is_aggregate_v<lbm::TRT<T>> && !std::is_default_constructible_v<lbm::TRT<T>>;
static_assert(trt_is_sealed<float> && trt_is_sealed<double>,
              "TRT must not be an aggregate nor default-constructible");

template <typename T>
ld eps_of() {
    return static_cast<ld>(std::numeric_limits<T>::epsilon());
}
const ld EPS_LD = std::numeric_limits<ld>::epsilon();

// ---------------------------------------------------------------------------
// Moments and equilibrium: bounds carried over from test_moments.cpp and
// test_equilibrium.cpp.
// ---------------------------------------------------------------------------
// Division and product u_d = j_d * inv_rho (test_moments.cpp):
//   |u~ - u| <= E (1 + |u|)/rho + K_u eps |u|,  E = absolute error of drho, j.
ld tol_u_from(ld E, ld eps, ld rho, ld absu) {
    return E * (1.0L + absu) / rho + K_u * eps * absu;
}

// Sum of magnitudes of the terms of f~eq_q (test_equilibrium.cpp), the scale of
// its rounding error: |err| <= K_e eps scale_e.
ld scale_e(int q, ld drho, const ld (&u)[L::D]) {
    const ld rho = 1.0L + drho;
    const ld inv_cs2 = L::inv_cs2<ld>();
    ld cu = 0.0L, uu = 0.0L;
    for (int d = 0; d < L::D; ++d) {
        cu += L::c(q, d) * u[d];
        uu += u[d] * u[d];
    }
    return L::w<ld>(q) * (std::abs(drho) + rho * (inv_cs2 * std::abs(cu) +
                                                  0.5L * inv_cs2 * inv_cs2 * cu * cu +
                                                  0.5L * inv_cs2 * uu));
}

// Propagation of an input perturbation through the equilibrium. With
//   g = drho + (1 + drho) P(u),   P = a cu + a^2 cu^2 / 2 - a uu / 2,  a = inv_cs2,
// a perturbation (alpha, b) with |drho'-drho| <= alpha, |b_d| <= beta_d gives
//   dg  = alpha + alpha P + (1 + drho) dP + alpha dP     (exact expansion),
//   dP  = a cb + a^2 cu cb + a^2 cb^2 / 2 - a (u.b) - a (b.b) / 2   (exact),
// with cb = c.b, so |dg| <= alpha (1 + |P| + |dP|) + rho |dP|, where
//   |P|  <= a|cu| + a^2 cu^2/2 + a uu/2          (sum of magnitudes),
//   |dP| <= a|cb| + a^2|cu||cb| + a^2 cb^2/2 + a|u.b| + a (b.b)/2,
//   |cb| <= sum_d |c_d| beta_d,  |u.b| <= sum_d |u_d| beta_d,  b.b <= sum_d beta_d^2.
// Multiplied by w_q. Nothing here is an estimate: it is the exact polynomial
// difference with each term replaced by its magnitude bound. The expansion
// point is the reference state; the difference to the true state is second
// order in the perturbation and covered by the headroom of K_e.
ld delta_eq(int q, ld drho, const ld (&u)[L::D], ld alpha, const ld (&beta)[L::D]) {
    const ld a = L::inv_cs2<ld>();
    const ld rho = 1.0L + drho;
    ld cu = 0.0L, uu = 0.0L, cb = 0.0L, ub = 0.0L, bb = 0.0L;
    for (int d = 0; d < L::D; ++d) {
        cu += L::c(q, d) * u[d];
        uu += u[d] * u[d];
        cb += std::abs(static_cast<ld>(L::c(q, d))) * beta[d];
        ub += std::abs(u[d]) * beta[d];
        bb += beta[d] * beta[d];
    }
    const ld absP = a * std::abs(cu) + 0.5L * a * a * cu * cu + 0.5L * a * uu;
    const ld dP = a * cb + a * a * std::abs(cu) * cb + 0.5L * a * a * cb * cb + a * ub +
                  0.5L * a * bb;
    return L::w<ld>(q) * (alpha * (1.0L + absP + dP) + rho * dP);
}

// ---------------------------------------------------------------------------
// Analysis of one input population set f (T-valued, exact as long double):
// long double reference moments and equilibrium, and per-direction bounds.
//
//   S      = sum_q |f~_q|
//   alpha_X = K_sum eps_X S                      bound on |drho_X - drho|
//   beta_X  = tol_u_from(alpha_X, eps_X, ...)    bound on |u_X - u|
// for X = T (the implementation) and X = ld (the reference), against the exact
// moments of the T-valued input.
//   E_T[q] : |feq_computed_q - Feq_q|, Feq the equilibrium at the exact moments:
//            delta_eq(alpha_T, beta_T) + K_e eps_T scale_e.
//   E_L[q] : |feq_ref_q - Feq_q|, the same with ld.
//   N[q]   : |f_q| + |feq_ref_q| + E_T[q] + E_L[q], a bound on |f_q| + |feq_computed_q|
//            (sum of magnitudes; it bounds |fneq_q| and every rounded operand).
// ---------------------------------------------------------------------------
struct Analysis {
    ld f[L::Q], feq[L::Q], E_T[L::Q], E_L[L::Q], N[L::Q], scale[L::Q];
    ld S, drho, rho, u[L::D];
};

template <typename T>
Analysis analyse(const T (&f)[L::Q]) {
    const ld eps_t = eps_of<T>();
    Analysis a;
    a.S = 0.0L;
    for (int q = 0; q < L::Q; ++q) {
        a.f[q] = f[q];
        a.S += std::abs(a.f[q]);
    }
    lbm::moments_shifted<L, ld>(a.f, a.drho, a.u);
    a.rho = 1.0L + a.drho;

    const ld alpha_t = K_sum * eps_t * a.S;
    const ld alpha_l = K_sum * EPS_LD * a.S;
    ld beta_t[L::D], beta_l[L::D];
    for (int d = 0; d < L::D; ++d) {
        beta_t[d] = tol_u_from(alpha_t, eps_t, a.rho, std::abs(a.u[d]));
        beta_l[d] = tol_u_from(alpha_l, EPS_LD, a.rho, std::abs(a.u[d]));
    }
    for (int q = 0; q < L::Q; ++q) {
        a.feq[q] = lbm::equilibrium_shifted<L, ld>(q, a.drho, a.u);
        a.scale[q] = scale_e(q, a.drho, a.u);
        a.E_T[q] = delta_eq(q, a.drho, a.u, alpha_t, beta_t) + K_e * eps_t * a.scale[q];
        a.E_L[q] = delta_eq(q, a.drho, a.u, alpha_l, beta_l) + K_e * EPS_LD * a.scale[q];
        a.N[q] = std::abs(a.f[q]) + std::abs(a.feq[q]) + a.E_T[q] + a.E_L[q];
    }
    return a;
}

// Bound on |out_computed_q - out*_q|, out* the collision in exact arithmetic
// with the exact equilibrium. With u = eps_T/2, Np = N_q + N_opp(q):
//   fneq_c  = fl(f - feq_c):   err <= E_T + u N                       (per direction)
//   fneq_pm = 0.5 fl(fneq_q +- fneq_opp): err <= 0.5 (E_q + E_o) + u Np
//             (0.5 (u N_q + u N_o) from the inputs, 0.5 u Np from the add;
//              the factor 0.5 is exact)
//   t1 = fl(wp fneq_plus), t2 = fl(wm fneq_minus):
//             err <= wp (Ep + 0.5 u Np) and wm (Em + 0.5 u Np), |fneq_pm| <= 0.5 Np
//   s = fl(f - t1), out = fl(s - t2): err <= u|s| + u|out|, with
//             |s| <= |f| + 0.5 wp Np, |out| <= |s| + 0.5 wm Np.
// Summing: (wp + wm) 0.5 (E_q + E_o) + u Np (2.5 wp + 2 wm) + 2 u |f_q|; the
// u-terms are multiplied by K_h. The rates are the stored T values (exact).
ld eout(const Analysis& a, int q, ld wp, ld wm, ld eps_t) {
    const int o = L::opp(q);
    const ld Np = a.N[q] + a.N[o];
    return (wp + wm) * 0.5L * (a.E_T[q] + a.E_T[o]) +
           K_h * 0.5L * eps_t * (Np * (2.5L * wp + 2.0L * wm) + 2.0L * std::abs(a.f[q]));
}

template <typename T>
void inputs_from_state(const State& s, T& drho, T (&u)[2], T (&f)[L::Q]) {
    drho = static_cast<T>(s.drho);
    u[0] = static_cast<T>(s.ux);
    u[1] = static_cast<T>(s.uy);
    for (int q = 0; q < L::Q; ++q) f[q] = lbm::equilibrium_shifted<L, T>(q, drho, u);
}

// Deterministic non-equilibrium perturbation of test_moments.cpp:
// p_q = 1e-3 w_q m_q, m_q = ((3q + 1) mod 7) - 3, not symmetric under q -> opp(q).
template <typename T>
void perturbed_input(const State& s, T (&f)[L::Q]) {
    T drho, u[2];
    inputs_from_state<T>(s, drho, u, f);
    for (int q = 0; q < L::Q; ++q) {
        const int m = ((3 * q + 1) % 7) - 3;
        f[q] = f[q] + T(1e-3) * L::template w<T>(q) * T(m);
    }
}

std::string describe(const State& s, const Params& p) {
    std::ostringstream o;
    o << "nu = " << p.nu << ", drho = " << s.drho << ", u = (" << s.ux << ", " << s.uy << ")";
    return o.str();
}

// ---------------------------------------------------------------------------
// Expected rates and their bounds, in long double from the double inputs.
//
// Reference (from nu): d = 3 nu (exact in ld), tau+ = d + 1/2, q = lambda/d,
// tau- = q + 1/2, omega = 1/tau. Its own error is at most 3 u_ld (rel), bounded
// by K_par eps_ld.
//
// Implementation, from nu (double, u_d = eps_d/2):
//   a = fl(3 nu): rel u;  tau+ = fl(a + 1/2): rel u;  so tau+ has rel error
//   <= 2u (all terms positive).  omega+ = fl(1/tau+): +u;  cast to T: +u_T.
//   => rel(omega+) <= 3 u_d + u_T  <= K_par eps_d + 0.5 eps_T.
//   d' = fl(tau+ - 1/2) has abs error <= 3 nu u + tau+ u (from a, tau+) + u d
//   <= 3 u tau+ (since d <= tau+), i.e. rel error 3u tau+/d.
//   q' = fl(lambda/d'): + u.  tau-' = fl(q' + 1/2): abs error
//   <= q (3u tau+/d + u) + u tau-  <= q (3u tau+/d) + 2 u tau-  (q <= tau-).
//   omega- = fl(1/tau-'): + u;  cast: + u_T.
//   => rel(omega-) <= 3u tau+/d + 3u + u_T <= K_par eps_d (tau+/d + 1) + 0.5 eps_T.
// From tau+ directly (tau+ exact input): d' = fl(tau+ - 1/2) rel u, q' rel 2u,
//   tau-' rel <= 3u (q' <= tau-), omega- rel <= 4u + u_T = 2 eps_d + 0.5 eps_T;
//   omega+ = fl(1/tau+): rel u + u_T <= eps_d + 0.5 eps_T.
// ---------------------------------------------------------------------------
struct Expected {
    ld wp, wm;        // reference rates
    ld tau_p, d, tau_m, q;
    ld rel_p, rel_m;  // relative bound on the stored rate vs the exact real rate
};

template <typename T>
Expected expected_from_viscosity(const Params& p) {
    Expected e;
    e.d = L::inv_cs2<ld>() * static_cast<ld>(p.nu);
    e.tau_p = e.d + 0.5L;
    e.q = static_cast<ld>(p.lambda) / e.d;
    e.tau_m = e.q + 0.5L;
    e.wp = 1.0L / e.tau_p;
    e.wm = 1.0L / e.tau_m;
    const ld eps_d = eps_of<double>();
    e.rel_p = K_par * eps_d + 0.5L * eps_of<T>();
    e.rel_m = K_par * eps_d * (e.tau_p / e.d + 1.0L) + 0.5L * eps_of<T>();
    return e;
}

template <typename T>
Expected expected_from_tau(double tau_plus, double lambda) {
    Expected e;
    e.tau_p = tau_plus;
    e.d = e.tau_p - 0.5L;
    e.q = static_cast<ld>(lambda) / e.d;
    e.tau_m = e.q + 0.5L;
    e.wp = 1.0L / e.tau_p;
    e.wm = 1.0L / e.tau_m;
    const ld eps_d = eps_of<double>();
    e.rel_p = 1.0L * eps_d + 0.5L * eps_of<T>();
    e.rel_m = 2.0L * eps_d + 0.5L * eps_of<T>();
    return e;
}

// Rate bound including the reference's own error.
ld rate_tol(ld w, ld rel) { return (rel + K_par * EPS_LD) * w; }

template <typename F>
void expect_rejected(F&& fn, const char* name, double value) {
    std::ostringstream shown;
    shown.precision(17);
    shown << value;
    try {
        fn();
        ADD_FAILURE() << "no exception for " << name << " = " << shown.str();
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find(name), std::string::npos) << what;
        EXPECT_NE(what.find(shown.str()), std::string::npos) << what;
    }
}

} // namespace

template <typename T>
class CollisionTRT : public ::testing::Test {
protected:
    using Trt = lbm::TRT<T>;
    static Trt make(const Params& p) { return Trt::template from_viscosity<L>(p.nu, p.lambda); }
};

using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CollisionTRT, PrecisionTypes);

// a. Parameters.
TYPED_TEST(CollisionTRT, ParametersFromViscosityAndTau) {
    using T = TypeParam;
    for (const Params& p : {P1, P2}) {
        const Expected e = expected_from_viscosity<T>(p);
        const auto trt = lbm::TRT<T>::template from_viscosity<L>(p.nu, p.lambda);
        EXPECT_LE(std::abs(static_cast<ld>(trt.omega_plus()) - e.wp), rate_tol(e.wp, e.rel_p))
            << "nu = " << p.nu;
        EXPECT_LE(std::abs(static_cast<ld>(trt.omega_minus()) - e.wm), rate_tol(e.wm, e.rel_m))
            << "nu = " << p.nu;

        // from_tau with its own documented rates.
        const double tau = 0.5 + 0.25 * (p.nu * 6.0 + 0.5);  // arbitrary tau > 1/2
        const Expected et = expected_from_tau<T>(tau, p.lambda);
        const auto tt = lbm::TRT<T>::from_tau(tau, p.lambda);
        EXPECT_LE(std::abs(static_cast<ld>(tt.omega_plus()) - et.wp), rate_tol(et.wp, et.rel_p))
            << "tau = " << tau;
        EXPECT_LE(std::abs(static_cast<ld>(tt.omega_minus()) - et.wm), rate_tol(et.wm, et.rel_m))
            << "tau = " << tau;

        // from_viscosity calls from_tau(nu * inv_cs2 + 0.5, lambda) and the test
        // forms tau_nu with the same double expression: bitwise identical rates.
        const double tau_nu = p.nu * L::inv_cs2<double>() + 0.5;
        const auto tv = lbm::TRT<T>::from_tau(tau_nu, p.lambda);
        EXPECT_EQ(tv.omega_plus(), trt.omega_plus());
        EXPECT_EQ(tv.omega_minus(), trt.omega_minus());
    }
}

// Lambda recovered from the stored rates. With stored omega = omega_exact (1 + r),
// |r| <= rho (the rel bound above), 1/omega_stored = tau / (1 + r) differs from tau
// by at most tau rho' with rho' = rho / (1 - rho). Then
//   d_rec = 1/omega+ - 1/2 has rel error a = rho'_p tau+/d,
//   q_rec = 1/omega- - 1/2 has rel error b = rho'_m tau-/q,
//   d_rec q_rec = lambda (1 + a)(1 + b)  =>  |err| <= lambda (a + b + a b).
// The long double evaluation adds, per rounding, u_ld of the quantity it
// produces: 1/omega (u), the subtraction (u tau, relative to d: u tau/d), the
// same for q, and the product (u): lambda u_ld (tau+/d + tau-/q + 3), bounded
// with u_ld -> eps_ld.
TYPED_TEST(CollisionTRT, LambdaRecoveredFromStoredRates) {
    using T = TypeParam;
    for (const Params& p : {P1, P2}) {
        const Expected e = expected_from_viscosity<T>(p);
        const auto trt = this->make(p);
        const ld lam = p.lambda;
        const ld rp = e.rel_p / (1.0L - e.rel_p);
        const ld rm = e.rel_m / (1.0L - e.rel_m);
        const ld a = rp * e.tau_p / e.d;
        const ld b = rm * e.tau_m / e.q;
        const ld tol = lam * (a + b + a * b) + lam * EPS_LD * (e.tau_p / e.d + e.tau_m / e.q + 3.0L);
        const ld rec = (1.0L / static_cast<ld>(trt.omega_plus()) - 0.5L) *
                       (1.0L / static_cast<ld>(trt.omega_minus()) - 0.5L);
        EXPECT_LE(std::abs(rec - lam), tol) << "nu = " << p.nu << ", lambda = " << p.lambda;
    }
}

// b. Rejection.
TYPED_TEST(CollisionTRT, RejectsInvalidParameters) {
    using T = TypeParam;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    for (double nu : {0.0, -1.0, nan, inf}) {
        EXPECT_THROW((lbm::TRT<T>::template from_viscosity<L>(nu, 0.25)), std::invalid_argument);
        expect_rejected([&] { lbm::TRT<T>::template from_viscosity<L>(nu, 0.25); }, "nu", nu);
    }
    for (double tau : {0.5, 0.4, nan}) {
        EXPECT_THROW(lbm::TRT<T>::from_tau(tau, 0.25), std::invalid_argument);
        expect_rejected([&] { lbm::TRT<T>::from_tau(tau, 0.25); }, "tau_plus", tau);
    }
    for (double lam : {0.0, -0.1, nan, inf}) {
        EXPECT_THROW((lbm::TRT<T>::template from_viscosity<L>(0.1, lam)), std::invalid_argument);
        EXPECT_THROW(lbm::TRT<T>::from_tau(0.8, lam), std::invalid_argument);
        expect_rejected([&] { lbm::TRT<T>::template from_viscosity<L>(0.1, lam); }, "lambda", lam);
        expect_rejected([&] { lbm::TRT<T>::from_tau(0.8, lam); }, "lambda", lam);
    }
    // nu = inf must not be reported as an invalid tau_plus.
    expect_rejected([&] { lbm::TRT<T>::template from_viscosity<L>(inf, 0.25); }, "nu", inf);
}

// Rates that are not in (0, 2) once stored as T. Each case is built so that
// the double value is valid but its image in T is not.
TEST(CollisionTRTRates, RejectsOmegaOutsideOpenIntervalAsStored) {
    // omega+ = 1/(1/2 + 1e-9) = 2 - 8e-9 rounds to 2 in float, but not in double.
    EXPECT_NO_THROW(lbm::TRT<double>::from_tau(0.5 + 1e-9, 0.25));
    expect_rejected([] { lbm::TRT<float>::from_tau(0.5 + 1e-9, 0.25); }, "omega_plus", 2.0);
    // omega+ underflows to 0 in float.
    expect_rejected([] { lbm::TRT<float>::from_tau(1e300, 0.25); }, "omega_plus", 0.0);
    // lambda tiny: tau- = lambda/d + 1/2 rounds to 1/2, omega- = 2.
    expect_rejected([] { lbm::TRT<double>::from_tau(1.0, 1e-30); }, "omega_minus", 2.0);
}

// c. Rest state.
TYPED_TEST(CollisionTRT, RestStateStaysExactlyZero) {
    using T = TypeParam;
    for (const Params& p : {P1, P2}) {
        T f[L::Q];
        for (int q = 0; q < L::Q; ++q) f[q] = T(0);
        this->make(p).template collide<L>(f);
        for (int q = 0; q < L::Q; ++q) EXPECT_EQ(f[q], T(0)) << "q = " << q;
    }
}

// d. Conservation on perturbed populations.
//
// Notation: wp, wm stored rates, u = eps_T/2, S = sum|f~|, N, scale from
// analyse(), t1_q = fl(wp fneq_plus_q), t2_q = fl(wm fneq_minus_q).
// out_q = f_q - t1_q - t2_q + r1_q + r2_q with |r1| <= u|s|, |r2| <= u|out|, so
//   sum_q c_qd (out_q - f_q) = -sum c t1 - sum c t2 + sum c (r1 + r2)     (c_q0 = 1 for density).
// Exact cancellations in the computed values: fneq_plus is bitwise symmetric
// and fneq_minus bitwise antisymmetric under q -> opp(q), hence
//   t1_q = t1_opp, t2_q = -t2_opp  =>  sum_q t2 = 0 and sum_q c_qd t1 = 0.
// Mass: remaining -sum t1 = -wp sum fp_c (1 + delta), and
//   sum fp_c = sum fn_c + R,        |R| <= u sum N       (adds, 0.5 u Np per q)
//   sum fn_c = sum f - sum feq_c + R', |R'| <= u sum N   (subtractions)
//   sum f - sum feq_c = (sum f - drho_c) + (drho_c - sum feq_c)
//        |.| <= K_sum eps S + K_e eps sum scale_e        (sum f - drho_c from the
//        moment sum; sum_q feq_c = drho_c up to the equilibrium rounding, since the
//        second-order equilibrium has the exact density moment drho)
//   |wp delta sum fp| <= wp u sum N.
//   => |d mass| <= wp (K_sum eps S + K_e eps sum scale + 3 u sum N) + roundings, where
//   the roundings are K_h u sum_q (2|f_q| + (wp + 0.5 wm) Np)  (|s| + |out| per q).
// Momentum (sum_q c_qd t1 = 0): -sum c t2 = -wm sum c fm_c (1 + delta) with
//   sum c fm_c = sum c fn_c + R, |R| <= u sum|c| N,
//   sum c fn_c = j_in - sum c feq_c + R', |R'| <= u sum|c| N,
//   sum c feq_c = rho~ u_c + (equilibrium rounding K_e eps sum|c| scale), where
//   rho~ = fl(1 + drho_c) is the density used by moments_shifted for inv_rho, so
//   rho~ u_c = j_c (1 + d2)(1 + d3), |rho~ u_c - j_c| <= eps |j_c| <= eps S, and
//   |j_c - j_in| <= K_sum eps S.
//   => |d j_d| <= wm ((K_sum + 1) eps S + K_e eps sum|c_d| scale + 3 u sum|c_d| N)
//                 + K_h u sum_q |c_qd| (2|f_q| + (wp + 0.5 wm) Np).
// The reference sums (long double, before and after) add K_sum eps_ld (S_in + S_out).
// All u-terms of the chain are multiplied by K_h.
template <typename T>
void check_conservation(const lbm::TRT<T>& trt, const Params& p) {
    const ld eps_t = eps_of<T>();
    const ld u_t = 0.5L * eps_t;
    for (const State& s : low_mach_states()) {
        T f[L::Q];
        perturbed_input<T>(s, f);
        const Analysis a = analyse<T>(f);
        const ld wp = trt.omega_plus(), wm = trt.omega_minus();

        T out[L::Q];
        for (int q = 0; q < L::Q; ++q) out[q] = f[q];
        trt.template collide<L>(out);

        ld S_out = 0.0L, sum_scale = 0.0L, sum_N = 0.0L, sum_round = 0.0L;
        ld sc[L::D] = {0.0L, 0.0L}, sN[L::D] = {0.0L, 0.0L}, sr[L::D] = {0.0L, 0.0L};
        ld drho_in = 0.0L, drho_out = 0.0L, j_in[L::D] = {0.0L, 0.0L}, j_out[L::D] = {0.0L, 0.0L};
        for (int q = 0; q < L::Q; ++q) {
            const ld Np = a.N[q] + a.N[L::opp(q)];
            const ld round_q = 2.0L * std::abs(a.f[q]) + (wp + 0.5L * wm) * Np;
            S_out += std::abs(static_cast<ld>(out[q]));
            sum_scale += a.scale[q];
            sum_N += a.N[q];
            sum_round += round_q;
            drho_in += a.f[q];
            drho_out += static_cast<ld>(out[q]);
            for (int d = 0; d < L::D; ++d) {
                const ld ac = std::abs(static_cast<ld>(L::c(q, d)));
                sc[d] += ac * a.scale[q];
                sN[d] += ac * a.N[q];
                sr[d] += ac * round_q;
                j_in[d] += L::c(q, d) * a.f[q];
                j_out[d] += L::c(q, d) * static_cast<ld>(out[q]);
            }
        }
        const ld ld_sums = K_sum * EPS_LD * (a.S + S_out);

        const ld tol_mass = wp * (K_sum * eps_t * a.S + K_e * eps_t * sum_scale +
                                  K_h * 3.0L * u_t * sum_N) +
                            K_h * u_t * sum_round + ld_sums;
        EXPECT_LE(std::abs(drho_out - drho_in), tol_mass) << "mass, " << describe(s, p);
        for (int d = 0; d < L::D; ++d) {
            const ld tol_j = wm * ((K_sum + 1.0L) * eps_t * a.S + K_e * eps_t * sc[d] +
                                   K_h * 3.0L * u_t * sN[d]) +
                             K_h * u_t * sr[d] + ld_sums;
            EXPECT_LE(std::abs(j_out[d] - j_in[d]), tol_j) << "momentum d = " << d << ", "
                                                           << describe(s, p);
        }
    }
}

TYPED_TEST(CollisionTRT, ConservesMassAndMomentumP1) {
    check_conservation<TypeParam>(this->make(P1), P1);
}
TYPED_TEST(CollisionTRT, ConservesMassAndMomentumP2) {
    check_conservation<TypeParam>(this->make(P2), P2);
}

// e. Pairwise relaxation (P1: omega+ != omega-).
//
// With F the exact equilibrium of the input's moments and out* the exact-arithmetic
// collision, out* - F has symmetric part (1 - wp) sym(f - F) and antisymmetric part
// (1 - wm) anti(f - F) identically (collision conserves the moments, so F is
// the equilibrium of the output too). The test replaces F by feq_ref and out* by
// out_computed:
//   |sym(out_c - out*)| <= 0.5 (eout_q + eout_o), likewise anti;
//   replacing F by feq_ref in both sides: difference wp sym(feq_ref - F)
//   (resp. wm anti(...)), bounded by w 0.5 (E_L_q + E_L_o);
//   long double evaluation of both sides: LHS 2 adds/subs: u_ld (|a_q| + |a_o|),
//   |a| <= |out| + |feq|; RHS 1.5 u_ld (|b_q| + |b_o|) + 0.5 u_ld Np (the (1 - w)
//   factor), |b| <= N; |out| <= |f| + 0.5 (wp + wm) Np:
//   K_h u_ld (Op + 3 Np), Op = sum over the pair of |f| + 0.5 (wp + wm) Np.
TYPED_TEST(CollisionTRT, PairwiseRelaxationP1) {
    using T = TypeParam;
    const ld eps_t = eps_of<T>();
    const auto trt = this->make(P1);
    const ld wp = trt.omega_plus(), wm = trt.omega_minus();
    ASSERT_GT(std::abs(wp - wm), 1.0L) << "P1 must have distinct rates";
    for (const State& s : low_mach_states()) {
        T f[L::Q];
        perturbed_input<T>(s, f);
        const Analysis a = analyse<T>(f);
        T out[L::Q];
        for (int q = 0; q < L::Q; ++q) out[q] = f[q];
        trt.template collide<L>(out);

        for (int q = 0; q < L::Q; ++q) {
            const int o = L::opp(q);
            const ld oq = out[q], oo = out[o];
            const ld sym_out = 0.5L * ((oq - a.feq[q]) + (oo - a.feq[o]));
            const ld anti_out = 0.5L * ((oq - a.feq[q]) - (oo - a.feq[o]));
            const ld sym_in = 0.5L * ((a.f[q] - a.feq[q]) + (a.f[o] - a.feq[o]));
            const ld anti_in = 0.5L * ((a.f[q] - a.feq[q]) - (a.f[o] - a.feq[o]));
            const ld Np = a.N[q] + a.N[o];
            const ld Op = std::abs(a.f[q]) + std::abs(a.f[o]) + (wp + wm) * Np;
            const ld ld_round = K_h * 0.5L * EPS_LD * (Op + 3.0L * Np);
            const ld eo = 0.5L * (eout(a, q, wp, wm, eps_t) + eout(a, o, wp, wm, eps_t));
            const ld el = 0.5L * (a.E_L[q] + a.E_L[o]);

            const ld err_sym = std::abs(sym_out - (1.0L - wp) * sym_in);
            const ld tol_sym = eo + wp * el + ld_round;
            EXPECT_LE(err_sym, tol_sym) << "sym, q = " << q << ", " << describe(s, P1);
            const ld err_anti = std::abs(anti_out - (1.0L - wm) * anti_in);
            const ld tol_anti = eo + wm * el + ld_round;
            EXPECT_LE(err_anti, tol_anti) << "anti, q = " << q << ", " << describe(s, P1);
        }
    }
}

// f. Fixed point.
//
// Input f_q = fl(F_state_q) differs from the exact equilibrium of the state by
// r_q, |r_q| <= K_e eps_T scale_e (rounding of the T evaluation; F_state is
// the exact equilibrium of the T-valued state). It is therefore not exactly the
// equilibrium of its own moments: the moments of f move by
//   alpha_r = sum_q rb_q,  J_d = sum_q |c_qd| rb_q,  rb_q = the bound on |r_q|,
// and |u' - u| = |j' - rho' u| / rho' <= (J_d + alpha_r |u_d|) / (rho - alpha_r) =: beta_r,d.
// Then the nonequilibrium part of f is
//   |fn*_q| <= B_q = rb_q + delta_eq(q, state, alpha_r, beta_r),
// and the exact collision moves f by at most (wp + wm) 0.5 (B_q + B_opp). The
// computed collision adds eout (analysis of the actual input).
TYPED_TEST(CollisionTRT, EquilibriumIsFixedPoint) {
    using T = TypeParam;
    const ld eps_t = eps_of<T>();
    for (const Params& p : {P1, P2}) {
        const auto trt = this->make(p);
        const ld wp = trt.omega_plus(), wm = trt.omega_minus();
        for (const State& s : low_mach_states()) {
            T drho, u[2], f[L::Q];
            inputs_from_state<T>(s, drho, u, f);
            const ld drho_s = drho;
            const ld u_s[L::D] = {u[0], u[1]};
            const ld rho_s = 1.0L + drho_s;

            ld rb[L::Q], alpha_r = 0.0L, J[L::D] = {0.0L, 0.0L};
            for (int q = 0; q < L::Q; ++q) {
                rb[q] = K_e * eps_t * scale_e(q, drho_s, u_s);
                alpha_r += rb[q];
                for (int d = 0; d < L::D; ++d)
                    J[d] += std::abs(static_cast<ld>(L::c(q, d))) * rb[q];
            }
            ld beta_r[L::D];
            for (int d = 0; d < L::D; ++d)
                beta_r[d] = (J[d] + alpha_r * std::abs(u_s[d])) / (rho_s - alpha_r);
            ld B[L::Q];
            for (int q = 0; q < L::Q; ++q)
                B[q] = rb[q] + delta_eq(q, drho_s, u_s, alpha_r, beta_r);

            const Analysis a = analyse<T>(f);
            T out[L::Q];
            for (int q = 0; q < L::Q; ++q) out[q] = f[q];
            trt.template collide<L>(out);
            for (int q = 0; q < L::Q; ++q) {
                const ld tol = (wp + wm) * 0.5L * (B[q] + B[L::opp(q)]) +
                               eout(a, q, wp, wm, eps_t);
                const ld err = std::abs(static_cast<ld>(out[q]) - static_cast<ld>(f[q]));
                EXPECT_LE(err, tol) << "q = " << q << ", " << describe(s, p)
                                    << ", err = " << err << ", tol = " << tol;
            }
        }
    }
}

// g. BGK reduction (P2, equal stored rates, asserted). Reference:
// out_ref = f - w (f - feq_ref), w = omega_plus,
// in long double. Against out* (exact TRT collision, exact F):
//   |out_c - out*| <= eout;
//   exact TRT equals exact BGK(wp) because the stored rates are equal;
//   BGK(wp) with F vs with feq_ref: wp E_L;
//   long double evaluation of the reference (a = f - feq: u|a|, b = w a: u|b|,
//   out = f - b: u|out|): |a| <= N, |b| <= wp N, |out| <= |f| + wp N, so
//   K_h u_ld (N + 2 wp N + |f|).
TYPED_TEST(CollisionTRT, ReducesToBGKWhenRatesAreEqual) {
    using T = TypeParam;
    const ld eps_t = eps_of<T>();
    const auto trt = this->make(P2);
    ASSERT_EQ(trt.omega_plus(), trt.omega_minus()) << "P2 must give equal rates";
    const ld wp = trt.omega_plus(), wm = trt.omega_minus();
    for (const State& s : low_mach_states()) {
        T f[L::Q];
        perturbed_input<T>(s, f);
        const Analysis a = analyse<T>(f);
        T out[L::Q];
        for (int q = 0; q < L::Q; ++q) out[q] = f[q];
        trt.template collide<L>(out);
        for (int q = 0; q < L::Q; ++q) {
            const ld ref = a.f[q] - wp * (a.f[q] - a.feq[q]);
            const ld tol = eout(a, q, wp, wm, eps_t) + wp * a.E_L[q] +
                           K_h * 0.5L * EPS_LD * (a.N[q] + 2.0L * wp * a.N[q] + std::abs(a.f[q]));
            const ld err = std::abs(static_cast<ld>(out[q]) - ref);
            EXPECT_LE(err, tol) << "q = " << q << ", " << describe(s, P2) << ", err = " << err
                                << ", tol = " << tol;
        }
    }
}
