#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "lbm/config.hpp"
#include "lbm/equilibrium.hpp"
#include "lbm/force.hpp"
#include "lbm/moments.hpp"

namespace lbm {

namespace detail {

// Host-only. Throws std::invalid_argument naming the parameter and its value.
[[noreturn]] inline void trt_reject(const char* name, double value, const char* requirement) {
    std::ostringstream msg;
    msg.precision(17);
    msg << "TRT: invalid " << name << " = " << value << " (" << requirement << ")";
    throw std::invalid_argument(msg.str());
}

// Storage of the force policy inside TRT: empty for NoForce, so that the
// unforced collision keeps its size and layout.
template <typename Force>
struct TrtForceStorage {
    LBM_HD constexpr explicit TrtForceStorage(const Force& force) : force_(force) {}
    Force force_;
};
template <>
struct TrtForceStorage<NoForce> {
    LBM_HD constexpr explicit TrtForceStorage(const NoForce&) {}
};

} // namespace detail

// Two-relaxation-time collision on shifted populations f~_q = f_q - w_q.
//
// The populations are split into a symmetric and an antisymmetric part under
// q -> opp(q), relaxed with omega_plus and omega_minus respectively:
//
//   fneq_q       = f~_q - f~eq_q            (= f_q - feq_q, since w_q cancels)
//   fneq_plus_q  = (fneq_q + fneq_opp(q)) / 2
//   fneq_minus_q = (fneq_q - fneq_opp(q)) / 2
//   out_q        = f~_q - omega_plus fneq_plus_q - omega_minus fneq_minus_q.
//
// The relaxation times are tau_plus = nu / cs2 + 1/2 and
// tau_minus = Lambda / (tau_plus - 1/2) + 1/2, with the magic parameter
// Lambda = (tau_plus - 1/2)(tau_minus - 1/2); omega = 1 / tau. T is the
// compute type.
//
// The Force policy selects the body force: NoForce (default) is the unforced
// collision, UniformForce<T, D> adds the Guo forcing term (see collide).
template <typename T, typename Force = NoForce>
class TRT : private detail::TrtForceStorage<Force> {
    static constexpr bool forced = !std::is_same_v<Force, NoForce>;

public:
    // Host-only named constructors, the only way to obtain a TRT. Parameters are computed in double and the
    // resulting rates are stored as T. They throw std::invalid_argument for
    // nu <= 0, tau_plus <= 1/2, lambda <= 0, non-finite input, or a rate that
    // is not in the open interval (0, 2) once stored as T.
    template <typename L, typename F = Force, std::enable_if_t<std::is_same_v<F, NoForce>, int> = 0>
    static TRT from_viscosity(double nu, double lambda) {
        return from_viscosity_impl<L>(nu, lambda, NoForce{});
    }

    template <typename L, typename F = Force, std::enable_if_t<!std::is_same_v<F, NoForce>, int> = 0>
    static TRT from_viscosity(double nu, double lambda, const Force& force) {
        return from_viscosity_impl<L>(nu, lambda, force);
    }

    template <typename F = Force, std::enable_if_t<std::is_same_v<F, NoForce>, int> = 0>
    static TRT from_tau(double tau_plus, double lambda) {
        return from_tau_impl(tau_plus, lambda, NoForce{});
    }

    template <typename F = Force, std::enable_if_t<!std::is_same_v<F, NoForce>, int> = 0>
    static TRT from_tau(double tau_plus, double lambda, const Force& force) {
        return from_tau_impl(tau_plus, lambda, force);
    }

    LBM_HD constexpr T omega_plus() const { return omega_plus_; }
    LBM_HD constexpr T omega_minus() const { return omega_minus_; }

    // In-place collision on shifted populations. The full f_q is never formed.
    // The output is written back only after every out_q has been computed.
    //
    // Force = NoForce: out_q = f_q - omega_plus fneq_plus_q - omega_minus fneq_minus_q.
    //
    // Force = UniformForce (Guo et al. 2002): the equilibrium is evaluated at
    // the half-force velocity u = (j + F/2) / rho (moments_shifted_forced) and
    //
    //   out_q = f_q - omega_plus fneq_plus_q - omega_minus fneq_minus_q
    //           + (1 - omega_plus/2) s_sym_q + (1 - omega_minus/2) s_anti_q,
    //
    // with s_anti (odd under q -> opp) and s_sym (even) from guo_source_parts.
    // In exact arithmetic the collision conserves mass and changes the momentum
    // by exactly F per step: sum_q out_q = sum_q f_q and
    // sum_q c_q (out_q - f_q) = F. (sum_q c_q fneq_q = j - rho u = -F/2 is
    // relaxed with omega_minus, giving +omega_minus F/2, and the antisymmetric
    // source contributes (1 - omega_minus/2) F; the sum is F.)
    template <typename L>
    LBM_HD void collide(T (&f)[L::Q]) const {
        T drho;
        T u[L::D];
        T Fv[L::D];
        if constexpr (forced) {
            static_assert(std::is_same_v<Force, UniformForce<T, L::D>>,
                          "TRT force policy must be UniformForce<T, L::D>");
            LBM_UNROLL
            for (int d = 0; d < L::D; ++d) Fv[d] = this->force_.F(d);
            moments_shifted_forced<L, T>(f, Fv, drho, u);
        } else {
            moments_shifted<L, T>(f, drho, u);
        }

        T fneq[L::Q];
        LBM_UNROLL
        for (int q = 0; q < L::Q; ++q) fneq[q] = f[q] - equilibrium_shifted<L, T>(q, drho, u);

        T out[L::Q];
        LBM_UNROLL
        for (int q = 0; q < L::Q; ++q) {
            const T a = fneq[q];
            const T b = fneq[L::opp(q)];
            const T fneq_plus = T(0.5) * (a + b);
            const T fneq_minus = T(0.5) * (a - b);
            if constexpr (forced) {
                T s_anti, s_sym;
                guo_source_parts<L, T>(q, u, Fv, s_anti, s_sym);
                out[q] = f[q] - omega_plus_ * fneq_plus - omega_minus_ * fneq_minus +
                         (T(1) - T(0.5) * omega_plus_) * s_sym +
                         (T(1) - T(0.5) * omega_minus_) * s_anti;
            } else {
                out[q] = f[q] - omega_plus_ * fneq_plus - omega_minus_ * fneq_minus;
            }
        }

        LBM_UNROLL
        for (int q = 0; q < L::Q; ++q) f[q] = out[q];
    }

private:
    template <typename L>
    static TRT from_viscosity_impl(double nu, double lambda, const Force& force) {
        if (!std::isfinite(nu) || !(nu > 0.0))
            detail::trt_reject("nu", nu, "must be finite and > 0");
        return from_tau_impl(nu * L::template inv_cs2<double>() + 0.5, lambda, force);
    }

    static TRT from_tau_impl(double tau_plus, double lambda, const Force& force) {
        if (!std::isfinite(tau_plus) || !(tau_plus > 0.5))
            detail::trt_reject("tau_plus", tau_plus, "must be finite and > 1/2");
        if (!std::isfinite(lambda) || !(lambda > 0.0))
            detail::trt_reject("lambda", lambda, "must be finite and > 0");

        const double tau_minus = lambda / (tau_plus - 0.5) + 0.5;
        const T omega_plus = static_cast<T>(1.0 / tau_plus);
        const T omega_minus = static_cast<T>(1.0 / tau_minus);
        if (!(omega_plus > T(0) && omega_plus < T(2)))
            detail::trt_reject("omega_plus", static_cast<double>(omega_plus),
                               "must be in (0, 2) as stored");
        if (!(omega_minus > T(0) && omega_minus < T(2)))
            detail::trt_reject("omega_minus", static_cast<double>(omega_minus),
                               "must be in (0, 2) as stored");
        return TRT(omega_plus, omega_minus, force);
    }

    LBM_HD constexpr TRT(T omega_plus, T omega_minus, const Force& force)
        : detail::TrtForceStorage<Force>(force), omega_plus_(omega_plus), omega_minus_(omega_minus) {}

    T omega_plus_;
    T omega_minus_;
};

// Passed by value to kernels.
static_assert(std::is_trivially_copyable_v<TRT<float>>);
static_assert(std::is_trivially_copyable_v<TRT<double>>);
static_assert(std::is_trivially_copyable_v<TRT<float, UniformForce<float, 2>>>);
static_assert(std::is_trivially_copyable_v<TRT<double, UniformForce<double, 2>>>);
static_assert(sizeof(TRT<float>) == 2 * sizeof(float));
static_assert(sizeof(TRT<double>) == 2 * sizeof(double));

} // namespace lbm
