// Tests of the Guo body force: source parts, forced moments and the forced TRT
// collision.
//
// Accuracy of the T-precision forced collision against a long double
// reference is deferred to a later task (the helpers it needs are in
// support/error_bounds.hpp). Here:
//   a. regression against the unforced collision (exact),
//   b. parity of the source parts (exact),
//   c, d. semantics of the scheme (source moments, mass and momentum change)
//         in long double, where rounding is negligible and bounded explicitly,
//   e. composition of the forced collision from its parts (exact),
//   f. factory validation.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "lbm/collision/trt.hpp"
#include "lbm/equilibrium.hpp"
#include "lbm/force.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "lbm/moments.hpp"
#include "support/error_bounds.hpp"
#include "support/fields.hpp"
#include "support/fp_environment.hpp"

using namespace lbmtest;

namespace {

template <typename T>
using Force = lbm::UniformForce<T, L::D>;

struct Params {
    double nu;
    double lambda;
};
constexpr Params P1{0.01, 3.0 / 16.0};  // omega+ != omega-
constexpr Params P2{1.0 / 6.0, 0.25};   // omega+ = omega- = 1

// Forces with both components nonzero.
constexpr double kF[][2] = {{1e-5, -2e-5}, {3e-4, 7e-4}};

template <typename T>
class ForceTest : public ::testing::Test {};
using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ForceTest, PrecisionTypes);

// ---------------------------------------------------------------------------
// Error-bound constants: K_src, K_acc, K_w, K_uF, K_chain, K_sum, K_e and K_div
// and the scales source_scale and scale_e are defined with their derivations in
// support/error_bounds.hpp. The long double tests use u = eps_ld / 2 = kULd.
// ---------------------------------------------------------------------------

} // namespace

// a. With F = 0 the forced collision is bitwise the unforced one.
TYPED_TEST(ForceTest, ZeroForceReproducesUnforced) {
    using T = TypeParam;
    for (const Params& p : {P1, P2}) {
        const double zero[2] = {0.0, 0.0};
        const auto forced = lbm::TRT<T, Force<T>>::template from_viscosity<L>(
            p.nu, p.lambda, Force<T>::create(zero));
        const auto plain = lbm::TRT<T>::template from_viscosity<L>(p.nu, p.lambda);
        ASSERT_EQ(forced.omega_plus(), plain.omega_plus());
        ASSERT_EQ(forced.omega_minus(), plain.omega_minus());
        for (const State& s : low_mach_states()) {
            T f1[L::Q], f2[L::Q];
            perturbed_input<T>(s, f1);
            for (int q = 0; q < L::Q; ++q) f2[q] = f1[q];
            forced.template collide<L>(f1);
            plain.template collide<L>(f2);
            for (int q = 0; q < L::Q; ++q)
                EXPECT_EQ(f1[q], f2[q]) << "q = " << q << ", drho = " << s.drho << ", u = ("
                                        << s.ux << ", " << s.uy << ")";
        }
    }
}

// b. Parity under q -> opp(q): s_anti odd, s_sym even, bitwise. Holds because
// negating all c components negates cu and cF exactly (the sums are formed in
// the same order), and the remaining operations are identical.
TYPED_TEST(ForceTest, SourceParity) {
    using T = TypeParam;
    for (const auto& fc : kF)
        for (const State& s : low_mach_states()) {
            const T u[2] = {static_cast<T>(s.ux), static_cast<T>(s.uy)};
            const T F[2] = {static_cast<T>(fc[0]), static_cast<T>(fc[1])};
            for (int q = 0; q < L::Q; ++q) {
                T a1, s1, a2, s2;
                lbm::guo_source_parts<L, T>(q, u, F, a1, s1);
                lbm::guo_source_parts<L, T>(L::opp(q), u, F, a2, s2);
                EXPECT_EQ(a2, -a1) << "q = " << q;
                EXPECT_EQ(s2, s1) << "q = " << q;
            }
        }
}

// c. Moments of the source (long double). Exact identities (a = 1/cs2,
// sum_q w = 1, sum_q w c_i c_j = cs2 delta_ij,
// sum_q w c_i c_j c_k c_l = cs2^2 (d_ij d_kl + d_ik d_jl + d_il d_jk), odd
// moments zero):
//   sum s_anti = 0,  sum s_anti c_a = F_a,  sum s_sym = 0,  sum s_sym c_a = 0,
//   sum s_sym c_a c_b = u_a F_b + u_b F_a.
// Bound for a sum with weights g_q in {|1|, |c_a|, |c_a c_b|} (exact factors):
//   (K_src + K_acc + K_w) u sum_q g_q W_q   [source roundings, accumulation,
//                                            weight representation]
// plus K_uF u (|u_a F_b| + |u_b F_a|) for the computed reference of the last
// identity.
TEST(ForceSource, MomentsInLongDouble) {
    for (const auto& fc : kF)
        for (const State& st : low_mach_states()) {
            const ld u[2] = {static_cast<ld>(st.ux), static_cast<ld>(st.uy)};
            const ld F[2] = {static_cast<ld>(fc[0]), static_cast<ld>(fc[1])};
            ld sa[L::Q], ss[L::Q], W[L::Q];
            for (int q = 0; q < L::Q; ++q) {
                lbm::guo_source_parts<L, ld>(q, u, F, sa[q], ss[q]);
                W[q] = source_scale(q, u, F);
            }
            const ld k = (K_src + K_acc + K_w) * kULd;

            ld m0a = 0, m0s = 0, W0 = 0;
            for (int q = 0; q < L::Q; ++q) {
                m0a += sa[q];
                m0s += ss[q];
                W0 += W[q];
            }
            EXPECT_LE(std::abs(m0a), k * W0);
            EXPECT_LE(std::abs(m0s), k * W0);

            for (int a = 0; a < L::D; ++a) {
                ld m1a = 0, m1s = 0, W1 = 0;
                for (int q = 0; q < L::Q; ++q) {
                    const ld c = L::c(q, a);
                    m1a += c * sa[q];
                    m1s += c * ss[q];
                    W1 += std::abs(c) * W[q];
                }
                EXPECT_LE(std::abs(m1a - F[a]), k * W1 + K_uF * kULd * std::abs(F[a])) << "a = " << a;
                EXPECT_LE(std::abs(m1s), k * W1) << "a = " << a;

                for (int b = 0; b < L::D; ++b) {
                    ld m2 = 0, W2 = 0;
                    for (int q = 0; q < L::Q; ++q) {
                        const ld c = static_cast<ld>(L::c(q, a)) * L::c(q, b);
                        m2 += c * ss[q];
                        W2 += std::abs(c) * W[q];
                    }
                    const ld ref = u[a] * F[b] + u[b] * F[a];
                    const ld ref_mag = std::abs(u[a] * F[b]) + std::abs(u[b] * F[a]);
                    EXPECT_LE(std::abs(m2 - ref), k * W2 + K_uF * kULd * ref_mag)
                        << "a = " << a << ", b = " << b;
                }
            }
        }
}

// d. Collision identities (long double): the forced collision conserves mass
// and changes the momentum by F (exact arithmetic; see TRT::collide).
//
// Derivation of the bound. Notation: omega_p/omega_m the rates, N_q =
// |f_q| + |feq_q| (feq at the computed drho_c, u_c), S = sum|f|,
// W_q = source_scale, M_q = |f_q| + (omega_p + omega_m)/2 (N_q + N_opp) + W_q
// (a bound on every partial result of the chain of out_q, using
// 0 < 1 - omega/2 < 1). The computed fneq_plus is bitwise even and
// fneq_minus bitwise odd, so sum_q c t1 = 0 and sum_q t2 = 0 exactly (t1, t2
// the rounded relaxation terms). Then
//   mass:     |sum (out - f)| <= omega_p |sum f - sum feq_c| + chain + source,
//     |sum f - drho_c| <= K_sum eps S (the sum in moments_shifted_forced),
//     |sum feq_c - drho_c| <= K_e eps sum scale_e + K_w u sum scale_e
//     (the density moment of the second-order equilibrium is exactly drho;
//     rounding of its evaluation and of its weights);
//   momentum: sum_q c fneq_q = j - sum c feq_c, and sum c feq_c = rho~ u_c
//     (rho~ = fl(1 + drho_c)), rho~ u_c = (j_c + F/2)(1 + delta),
//     |delta| <= K_div u, hence sum c fneq_q = -F/2 up to
//     (K_sum eps S + K_e eps sum|c| scale + K_w u sum|c| scale +
//      K_div u (S + |F|/2)); this is relaxed with omega_m, giving
//     +omega_m F/2, and sum c s_anti = F gives (1 - omega_m/2) F exactly.
//   chain:  K_chain u sum_q g_q (M_q + N_q + N_opp)
//   source: (K_src + K_w) u sum_q g_q W_q       (g_q = 1 for mass, |c_qd| for
//     momentum; the sums sum_q s = 0 and sum_q c s_sym = 0 hold in exact
//     arithmetic and the roundings of the sources are counted here)
//   test:   K_sum eps_ld (S_in + S_out)  for the ld sums formed by this test.
TEST(ForceCollide, ConservesMassAndMomentumChangeIsF) {
    using Fld = Force<ld>;
    for (const Params& p : {P1, P2})
        for (const auto& fc : kF) {
            const double fcd[2] = {fc[0], fc[1]};
            const auto trt = lbm::TRT<ld, Fld>::from_viscosity<L>(p.nu, p.lambda, Fld::create(fcd));
            const ld wp = trt.omega_plus(), wm = trt.omega_minus();
            // Precondition of the two parameter sets: distinct rates for P1,
            // equal rates (both 1) for P2.
            if (p.nu == P1.nu) {
                ASSERT_NE(wp, wm);
            } else {
                ASSERT_EQ(wp, wm);
            }
            const ld F[2] = {static_cast<ld>(fc[0]), static_cast<ld>(fc[1])};

            for (const State& st : low_mach_states()) {
                ld f[L::Q];
                perturbed_input<ld>(st, f);

                ld drho, u[L::D];
                lbm::moments_shifted_forced<L, ld>(f, F, drho, u);

                ld S = 0, sum_scale = 0, sum_chain = 0, sum_W = 0;
                ld sc[L::D] = {0, 0}, schain[L::D] = {0, 0}, sW[L::D] = {0, 0};
                ld N[L::Q];
                for (int q = 0; q < L::Q; ++q)
                    N[q] = std::abs(f[q]) + std::abs(lbm::equilibrium_shifted<L, ld>(q, drho, u));
                ld mass_in = 0, j_in[L::D] = {0, 0};
                for (int q = 0; q < L::Q; ++q) {
                    const ld Wq = source_scale(q, u, F);
                    const ld Np = N[q] + N[L::opp(q)];
                    const ld Mq = std::abs(f[q]) + 0.5L * (wp + wm) * Np + Wq;
                    const ld chain_q = Mq + Np;
                    S += std::abs(f[q]);
                    sum_scale += scale_e(q, drho, u);
                    sum_chain += chain_q;
                    sum_W += Wq;
                    mass_in += f[q];
                    for (int d = 0; d < L::D; ++d) {
                        const ld ac = std::abs(static_cast<ld>(L::c(q, d)));
                        sc[d] += ac * scale_e(q, drho, u);
                        schain[d] += ac * chain_q;
                        sW[d] += ac * Wq;
                        j_in[d] += L::c(q, d) * f[q];
                    }
                }

                ld out[L::Q];
                for (int q = 0; q < L::Q; ++q) out[q] = f[q];
                trt.collide<L>(out);

                ld S_out = 0, mass_out = 0, j_out[L::D] = {0, 0};
                for (int q = 0; q < L::Q; ++q) {
                    S_out += std::abs(out[q]);
                    mass_out += out[q];
                    for (int d = 0; d < L::D; ++d) j_out[d] += L::c(q, d) * out[q];
                }
                const ld test_sums = K_sum * kEpsLd * (S + S_out);

                const ld tol_mass = wp * (K_sum * kEpsLd * S + (K_e * kEpsLd + K_w * kULd) * sum_scale) +
                                    K_chain * kULd * sum_chain + (K_src + K_w) * kULd * sum_W + test_sums;
                EXPECT_LE(std::abs(mass_out - mass_in), tol_mass)
                    << "mass, nu = " << p.nu << ", drho = " << st.drho;

                for (int d = 0; d < L::D; ++d) {
                    const ld tol_j = wm * (K_sum * kEpsLd * S + (K_e * kEpsLd + K_w * kULd) * sc[d] +
                                           K_div * kULd * (S + 0.5L * std::abs(F[d]))) +
                                     K_chain * kULd * schain[d] + (K_src + K_w) * kULd * sW[d] +
                                     test_sums;
                    EXPECT_LE(std::abs((j_out[d] - j_in[d]) - F[d]), tol_j)
                        << "momentum d = " << d << ", nu = " << p.nu << ", drho = " << st.drho
                        << ", u = (" << st.ux << ", " << st.uy << ")";
                }
            }
        }
}

// e. The forced collision is exactly the composition of its parts, in the
// same operation order.
TYPED_TEST(ForceTest, CompositionFromParts) {
    using T = TypeParam;
    for (const auto& fc : kF) {
        const double fcd[2] = {fc[0], fc[1]};
        const auto trt =
            lbm::TRT<T, Force<T>>::template from_viscosity<L>(P1.nu, P1.lambda, Force<T>::create(fcd));
        ASSERT_NE(trt.omega_plus(), trt.omega_minus());
        const auto plain = lbm::TRT<T>::template from_viscosity<L>(P1.nu, P1.lambda);
        const T F[2] = {static_cast<T>(fc[0]), static_cast<T>(fc[1])};

        for (const State& s : low_mach_states()) {
            T f[L::Q];
            perturbed_input<T>(s, f);

            T drho, u[L::D];
            lbm::moments_shifted_forced<L, T>(f, F, drho, u);
            T fneq[L::Q];
            for (int q = 0; q < L::Q; ++q) fneq[q] = f[q] - lbm::equilibrium_shifted<L, T>(q, drho, u);
            T expected[L::Q];
            for (int q = 0; q < L::Q; ++q) {
                const T fp = T(0.5) * (fneq[q] + fneq[L::opp(q)]);
                const T fm = T(0.5) * (fneq[q] - fneq[L::opp(q)]);
                T s_anti, s_sym;
                lbm::guo_source_parts<L, T>(q, u, F, s_anti, s_sym);
                expected[q] = f[q] - trt.omega_plus() * fp - trt.omega_minus() * fm +
                              (T(1) - T(0.5) * trt.omega_plus()) * s_sym +
                              (T(1) - T(0.5) * trt.omega_minus()) * s_anti;
            }

            T out[L::Q], unforced[L::Q];
            for (int q = 0; q < L::Q; ++q) out[q] = unforced[q] = f[q];
            trt.template collide<L>(out);
            plain.template collide<L>(unforced);

            bool differs = false;
            for (int q = 0; q < L::Q; ++q) {
                EXPECT_EQ(out[q], expected[q]) << "q = " << q << ", drho = " << s.drho << ", u = ("
                                               << s.ux << ", " << s.uy << ")";
                differs = differs || (out[q] != unforced[q]);
            }
            // Guard: the force must actually change the result.
            EXPECT_TRUE(differs);
        }
    }
}

// g. Second moment of the collision with a body force, independent of the
// code's expression (long double). With g_ab(q) = c_qa c_qb (even under
// q -> opp(q)) and fneq_q = f_q - feq_q:
//
//   sum_q g (out_q - f_q)
//     = -wp sum g fneq_plus - wm sum g fneq_minus
//       + (1 - wp/2) sum g s_sym + (1 - wm/2) sum g s_anti.
//
// The odd parts do not contribute: fneq_minus_q = (fneq_q - fneq_opp(q))/2 and
// s_anti_q are odd under q -> opp(q) while g is even, so the pairs q, opp(q)
// cancel in sum g fneq_minus and sum g s_anti (for q = opp(q) = 0, c = 0 and
// g = 0). Since g is even, sum g fneq_plus = sum g fneq. The even source has
// the lattice moment sum g s_sym = u_a F_b + u_b F_a (test c). Hence
//
//   sum_q c_a c_b (out_q - f_q)
//     = -wp Pi_ab + (1 - wp/2) (u_a F_b + u_b F_a),   Pi_ab = sum c_a c_b fneq_q,
//
// with feq at the Guo velocity u = (j + F/2)/rho. Only wp enters, so with
// wp != wm (asserted) a wrong prefactor of s_sym, e.g. (1 - wm/2), or a
// missing s_sym changes the left side by far more than the bound.
//
// The reference uses the density and velocity that collide() computes
// (moments_shifted_forced, verified in test_moments.cpp and test d) and the
// same feq values and the same coefficient fl(1 - wp/2): the identity holds
// for any (drho, u, feq) in exact arithmetic, so their rounding errors do not
// enter, and no property of feq is used.
//
// Bound (first order in u = eps_ld/2; g_q in {0, 1} exact; sums over q):
//   chain of out_q:   K_chain u sum g (M_q + N_q + N_opp)       (as in test d)
//   source rounding:  (K_src + K_w) u sum g W_q                  (as in test d)
//   test arithmetic:  (K_acc + 1) u sum g (|out_q| + |f_q|)
//                     [one subtraction per term and the accumulation]
//   reference Pi:     (K_acc + 2) u wp sum g |fneq_q|
//                     [accumulation, product wp * Pi, final sum]
//   reference u F:    (K_uF + 2) u c1 (|u_a F_b| + |u_b F_a|)
//                     [3 roundings in u_a F_b + u_b F_a, product by c1, final
//                      sum], c1 = fl(1 - wp/2).
constexpr ld K_tst = K_acc + 1.0L;
constexpr ld K_refPi = K_acc + 2.0L;
constexpr ld K_refF = K_uF + 2.0L;
static_assert(K_tst <= 16.0L && K_refPi <= 16.0L && K_refF <= 16.0L,
              "every factor must not exceed 16");

TEST(ForceCollide, SecondMomentIncludesSymmetricSource) {
    using Fld = Force<ld>;
    int checked_nonzero = 0;
    for (const auto& fc : kF) {
        const double fcd[2] = {fc[0], fc[1]};
        ASSERT_NE(fc[0], 0.0);
        ASSERT_NE(fc[1], 0.0);
        const auto trt = lbm::TRT<ld, Fld>::from_viscosity<L>(P1.nu, P1.lambda, Fld::create(fcd));
        const ld wp = trt.omega_plus(), wm = trt.omega_minus();
        ASSERT_NE(wp, wm);
        const ld c1 = 1.0L - 0.5L * wp;
        const ld F[2] = {static_cast<ld>(fc[0]), static_cast<ld>(fc[1])};

        for (const State& st : low_mach_states()) {
            ld f[L::Q];
            perturbed_input<ld>(st, f);
            ld drho, u[L::D];
            lbm::moments_shifted_forced<L, ld>(f, F, drho, u);

            ld fneq[L::Q], N[L::Q];
            for (int q = 0; q < L::Q; ++q) {
                const ld feq = lbm::equilibrium_shifted<L, ld>(q, drho, u);
                fneq[q] = f[q] - feq;
                N[q] = std::abs(f[q]) + std::abs(feq);
            }

            ld out[L::Q];
            for (int q = 0; q < L::Q; ++q) out[q] = f[q];
            trt.collide<L>(out);

            for (int a = 0; a < L::D; ++a)
                for (int b = 0; b < L::D; ++b) {
                    ld lhs = 0, pi = 0, sum_fneq = 0, sum_chain = 0, sum_W = 0, sum_io = 0;
                    for (int q = 0; q < L::Q; ++q) {
                        const ld g = static_cast<ld>(L::c(q, a)) * L::c(q, b);
                        const ld Wq = source_scale(q, u, F);
                        const ld Np = N[q] + N[L::opp(q)];
                        const ld Mq = std::abs(f[q]) + 0.5L * (wp + wm) * Np + Wq;
                        lhs += g * (out[q] - f[q]);
                        pi += g * fneq[q];
                        sum_fneq += std::abs(g) * std::abs(fneq[q]);
                        sum_chain += std::abs(g) * (Mq + Np);
                        sum_W += std::abs(g) * Wq;
                        sum_io += std::abs(g) * (std::abs(out[q]) + std::abs(f[q]));
                    }
                    const ld uF = u[a] * F[b] + u[b] * F[a];
                    const ld uF_mag = std::abs(u[a] * F[b]) + std::abs(u[b] * F[a]);
                    const ld rhs = -wp * pi + c1 * uF;

                    const ld tol = kULd * (K_chain * sum_chain + (K_src + K_w) * sum_W +
                                           K_tst * sum_io + K_refPi * wp * sum_fneq +
                                           K_refF * c1 * uF_mag);
                    EXPECT_LE(std::abs(lhs - rhs), tol)
                        << "a = " << a << ", b = " << b << ", F = (" << fc[0] << ", " << fc[1]
                        << "), u = (" << st.ux << ", " << st.uy << ")";

                    // Precondition: the source term changes the left side by more than
                    // the bound, and the prefactor (1 - wm/2) would change it by more
                    // than the bound too, so the test can fail for both mutations.
                    if (uF != 0.0L) {
                        ++checked_nonzero;
                        ASSERT_GT(0.5L * std::abs(wp - wm) * std::abs(uF), tol);
                        ASSERT_GT(c1 * std::abs(uF), tol);
                    }
                }
        }
    }
    ASSERT_GT(checked_nonzero, 0);
}

// f. Factory rejection names the component.
TYPED_TEST(ForceTest, FactoryRejectsNonFinite) {
    using T = TypeParam;
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double bad[][2] = {{nan, 1.0}, {1.0, nan}, {inf, 0.0}, {0.0, -inf}};
    const char* name[] = {"F[0]", "F[1]", "F[0]", "F[1]"};
    for (int i = 0; i < 4; ++i) {
        try {
            (void)Force<T>::create(bad[i]);
            FAIL() << "expected std::invalid_argument, case " << i;
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(name[i]), std::string::npos) << e.what();
        }
    }
    const double ok[2] = {1e-5, -2e-5};
    const Force<T> force = Force<T>::create(ok);
    EXPECT_EQ(force.F(0), static_cast<T>(1e-5));
    EXPECT_EQ(force.F(1), static_cast<T>(-2e-5));
}

// A finite double that overflows float is rejected for T = float.
TEST(ForceFactory, RejectsValueNotFiniteInStorageType) {
    const double big[2] = {0.0, 1e300};
    EXPECT_THROW((void)Force<float>::create(big), std::invalid_argument);
    EXPECT_NO_THROW((void)Force<double>::create(big));
}

// The range check happens before the cast (the cast of an out-of-range double
// to float is undefined behaviour) and names the offending component.
TEST(ForceFactory, RangeCheckAtLargestFiniteValue) {
    const double fmax = static_cast<double>(std::numeric_limits<float>::max());
    const double at_max[2] = {fmax, -fmax};
    const Force<float> force = Force<float>::create(at_max);
    EXPECT_EQ(force.F(0), std::numeric_limits<float>::max());
    EXPECT_EQ(force.F(1), -std::numeric_limits<float>::max());

    const double above[2] = {0.0, -std::nextafter(fmax, 1e300)};
    ASSERT_GT(std::fabs(above[1]), fmax);
    try {
        (void)Force<float>::create(above);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("F[1]"), std::string::npos) << e.what();
    }
}
