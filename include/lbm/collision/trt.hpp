#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "lbm/config.hpp"
#include "lbm/equilibrium.hpp"
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
template <typename T>
class TRT {
public:
    // Host-only named constructors, the only way to obtain a TRT. Parameters are computed in double and the
    // resulting rates are stored as T. They throw std::invalid_argument for
    // nu <= 0, tau_plus <= 1/2, lambda <= 0, non-finite input, or a rate that
    // is not in the open interval (0, 2) once stored as T.
    template <typename L>
    static TRT from_viscosity(double nu, double lambda) {
        if (!std::isfinite(nu) || !(nu > 0.0))
            detail::trt_reject("nu", nu, "must be finite and > 0");
        return from_tau(nu * L::template inv_cs2<double>() + 0.5, lambda);
    }

    static TRT from_tau(double tau_plus, double lambda) {
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
        return TRT(omega_plus, omega_minus);
    }

    LBM_HD constexpr T omega_plus() const { return omega_plus_; }
    LBM_HD constexpr T omega_minus() const { return omega_minus_; }

    // In-place collision on shifted populations. The full f_q is never formed.
    // The output is written back only after every out_q has been computed.
    template <typename L>
    LBM_HD void collide(T (&f)[L::Q]) const {
        T drho;
        T u[L::D];
        moments_shifted<L, T>(f, drho, u);

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
            out[q] = f[q] - omega_plus_ * fneq_plus - omega_minus_ * fneq_minus;
        }

        LBM_UNROLL
        for (int q = 0; q < L::Q; ++q) f[q] = out[q];
    }

private:
    LBM_HD constexpr TRT(T omega_plus, T omega_minus)
        : omega_plus_(omega_plus), omega_minus_(omega_minus) {}

    T omega_plus_;
    T omega_minus_;
};

// Passed by value to kernels.
static_assert(std::is_trivially_copyable_v<TRT<float>>);
static_assert(std::is_trivially_copyable_v<TRT<double>>);

} // namespace lbm
