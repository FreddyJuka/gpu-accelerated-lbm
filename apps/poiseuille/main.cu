// Exploratory measurement: steady body-force-driven channel flow (Poiseuille)
// against the analytical profile, for several widths, viscosities and magic
// parameters, on the GPU. TRT with Guo forcing, halfway bounce-back walls at
// y = 0 and y = ny - 1.
//
// Options (default: double compute, storage the same as compute, i.e. the
// reference configuration):
//   --compute C   double | float              type of the collision arithmetic
//   --storage S   same | float | p16          type of the population buffers:
//                 same   the compute type (double/double or float/float)
//                 float  binary32 buffers (with --compute double: mixed precision)
//                 p16    16-bit buffers, lbm::P16 (requires --compute float)
//
// Velocity measurement. The stored populations are post-collision. The Guo
// half-force velocity u = (j + F/2) / rho is defined on the collision input,
// i.e. the populations after the gather (streaming + bounce-back). The
// velocity is therefore obtained by running the gather of the production node
// function (pull_stream_collide with an identity collision) on the stored
// populations and applying moments_shifted_forced to the result, in the
// compute type. As a diagnostic, the relative L2 error of
// moments_shifted_forced applied directly to the stored post-collision
// populations is also printed ("l2_stored").
//
// Reported per case: l2 and linf are the errors against the plain parabola
// u_a (relative to its L2 norm and to its maximum). err_slip is the maximum
// error against u_a + du, with the TRT slip du = F_x/(2 nu) (16 Lambda - 3)/12,
// relative to u_max = 0.01, evaluated in long double with the nu and Lambda
// effectively stored in the collision object (nu_eff = (1/omega_plus - 1/2)/3,
// Lambda_eff = (1/omega_plus - 1/2)(1/omega_minus - 1/2)); slip is du/u_max.
// max_uy and max_drho are maxima over the fluid nodes and mean_drho is the
// mean of drho over the fluid nodes (the initial state is at rest with
// drho = 0, so this is the drift of the mean density).
//
// Stopping rule: the change of u_x over 1000 steps is at most 1e-14 u_max in
// double, and the same number of eps (1e-14 u_max eps_float / eps_double) with
// float compute; at most 5e6 steps.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "lbm/collision/trt.hpp"
#include "lbm/cpu/step.hpp"
#include "lbm/cuda/check.hpp"
#include "lbm/cuda/device_buffer.hpp"
#include "lbm/cuda/step.cuh"
#include "lbm/force.hpp"
#include "lbm/grid.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "lbm/moments.hpp"
#include "lbm/propagation/ab_pull.hpp"
#include "lbm/storage.hpp"

namespace {

using L = lbm::D2Q9;
using Grid = lbm::Grid2D<L>;
constexpr double kUmax = 0.01;
constexpr int kCheckEvery = 1000;
constexpr long kStepCap = 5000000;
constexpr int kNx = 4;

// Stopping threshold on the change of u_x: 1e-14 u_max in double and the same
// number of eps in float (the ratio of the epsilons is a power of two, so the
// double value is exactly 1e-14 u_max).
template <typename Compute>
constexpr double kChangeTol =
    1e-14 * kUmax *
    (static_cast<double>(std::numeric_limits<Compute>::epsilon()) / std::numeric_limits<double>::epsilon());
static_assert(kChangeTol<double> == 1e-14 * kUmax);

// Identity collision in the compute type: the node function then performs the
// gather only, converting each population with load and back with store, which
// returns the stored value unchanged (store(load(s)) == s).
template <typename Compute>
struct IdentityCollision {
    using compute_type = Compute;
    template <typename Lat>
    LBM_HD void collide(Compute (&)[Lat::Q]) const {}
};

template <typename Compute>
struct Fields {
    std::vector<Compute> ux, uy, drho;  // per node, index y * nx + x
};

// Macroscopic fields of the stored (post-collision) populations. If gather is
// true, the gather is applied first (see the file comment).
template <typename Storage, typename Compute>
Fields<Compute> fields_from(const Grid& g, const std::vector<lbm::NodeType>& flags,
                            const std::vector<Storage>& stored,
                            const lbm::UniformForce<Compute, 2>& force, bool gather) {
    using Conversion = lbm::StorageConversion<Storage, Compute>;
    std::vector<Storage> work = stored;
    if (gather) {
        for (int y = 0; y < g.ny(); ++y)
            for (int x = 0; x < g.nx(); ++x)
                lbm::pull_stream_collide<L, Storage>(g, flags.data(), stored.data(), work.data(),
                                                     IdentityCollision<Compute>{}, x, y);
    }
    Fields<Compute> r;
    const std::size_t n = static_cast<std::size_t>(g.num_nodes());
    r.ux.assign(n, Compute(0));
    r.uy.assign(n, Compute(0));
    r.drho.assign(n, Compute(0));
    const Compute F[2] = {force.F(0), force.F(1)};
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x) {
            if (flags[static_cast<std::size_t>(g.node(x, y))] == lbm::NodeType::Wall) continue;
            Compute f[L::Q], u[2], drho;
            for (int q = 0; q < L::Q; ++q)
                f[q] = Conversion::load(work[static_cast<std::size_t>(g.idx(q, x, y))]);
            lbm::moments_shifted_forced<L, Compute>(f, F, drho, u);
            const std::size_t i = static_cast<std::size_t>(g.node(x, y));
            r.ux[i] = u[0];
            r.uy[i] = u[1];
            r.drho[i] = drho;
        }
    return r;
}

struct Result {
    int H;
    double nu, lambda;
    long steps;
    double change, l2, linf, l2_stored, max_uy, max_drho;
    double slip, err_slip, mean_drho;
    bool capped;
};

template <typename Storage, typename Compute>
Result run_case(int H, double nu, double lambda) {
    using Force = lbm::UniformForce<Compute, 2>;
    using Collision = lbm::TRT<Compute, Force>;
    const int ny = H + 2;
    const Grid g(kNx, ny);
    const int n = L::Q * g.num_nodes();

    std::vector<lbm::NodeType> flags(static_cast<std::size_t>(g.num_nodes()), lbm::NodeType::Fluid);
    for (int x = 0; x < kNx; ++x) {
        flags[static_cast<std::size_t>(g.node(x, 0))] = lbm::NodeType::Wall;
        flags[static_cast<std::size_t>(g.node(x, ny - 1))] = lbm::NodeType::Wall;
    }

    const double Fx = 8.0 * nu * kUmax / (static_cast<double>(H) * H);
    const double fc[2] = {Fx, 0.0};
    const Force force = Force::create(fc);
    const Collision coll = Collision::template from_viscosity<L>(nu, lambda, force);

    // Rest state: every stored population is the value-initialised Storage
    // (0.0, or the all-zero P16 pattern, which is +0).
    const std::vector<Storage> zeros(static_cast<std::size_t>(n), Storage{});
    lbm::DeviceBuffer<Storage> da(n), db(n);
    lbm::DeviceBuffer<lbm::NodeType> d_flags(g.num_nodes());
    da.copy_from_host(zeros.data(), n);
    db.copy_from_host(zeros.data(), n);
    d_flags.copy_from_host(flags.data(), g.num_nodes());

    std::vector<Storage> host(static_cast<std::size_t>(n));
    std::vector<Compute> ux_prev(static_cast<std::size_t>(g.num_nodes()), Compute(0));  // rest

    long steps = 0;
    double change = 0.0;
    bool capped = false;
    Fields<Compute> cur;
    for (;;) {
        for (int k = 0; k < kCheckEvery; ++k) {
            lbm::step_gpu<L, Storage>(g, d_flags.data(), da.data(), db.data(), coll);
            std::swap(da, db);
        }
        steps += kCheckEvery;
        LBM_CUDA_CHECK(cudaDeviceSynchronize());
        da.copy_to_host(host.data(), n);
        cur = fields_from<Storage, Compute>(g, flags, host, force, true);
        change = 0.0;
        for (std::size_t i = 0; i < cur.ux.size(); ++i)
            if (flags[i] == lbm::NodeType::Fluid)
                change = std::max(change, static_cast<double>(std::abs(cur.ux[i] - ux_prev[i])));
        ux_prev = cur.ux;
        if (change <= kChangeTol<Compute>) break;
        if (steps >= kStepCap) {
            capped = true;
            break;
        }
    }

    const Fields<Compute> stored = fields_from<Storage, Compute>(g, flags, host, force, false);

    // Errors against u_a(y) = Fx / (2 nu) (y - 0.5)(ny - 1.5 - y), nominal
    // parameters, in double.
    auto analytic = [&](int y) { return Fx / (2.0 * nu) * (y - 0.5) * (ny - 1.5 - y); };
    // Reference with the slip, in long double, from the stored force and rates.
    const long double tp = 1.0L / static_cast<long double>(coll.omega_plus());
    const long double tm = 1.0L / static_cast<long double>(coll.omega_minus());
    const long double nu_eff = (tp - 0.5L) / 3.0L;
    const long double lambda_eff = (tp - 0.5L) * (tm - 0.5L);
    const long double amp = static_cast<long double>(force.F(0)) / (2.0L * nu_eff);
    const long double du = amp * (16.0L * lambda_eff - 3.0L) / 12.0L;
    auto analytic_slip = [&](int y) {
        return amp * (static_cast<long double>(y) - 0.5L) * (ny - 1.5L - static_cast<long double>(y)) + du;
    };

    double num = 0.0, den = 0.0, num_s = 0.0, linf = 0.0, umax_a = 0.0, max_uy = 0.0, max_drho = 0.0;
    long double err_slip = 0.0L, sum_drho = 0.0L;
    for (int y = 1; y < ny - 1; ++y)
        for (int x = 0; x < kNx; ++x) {
            const std::size_t i = static_cast<std::size_t>(g.node(x, y));
            const double ua = analytic(y);
            const double e = static_cast<double>(cur.ux[i]) - ua;
            num += e * e;
            den += ua * ua;
            const double es = static_cast<double>(stored.ux[i]) - ua;
            num_s += es * es;
            linf = std::max(linf, std::abs(e));
            umax_a = std::max(umax_a, ua);
            max_uy = std::max(max_uy, static_cast<double>(std::abs(cur.uy[i])));
            max_drho = std::max(max_drho, static_cast<double>(std::abs(cur.drho[i])));
            err_slip = std::max(err_slip, std::abs(static_cast<long double>(cur.ux[i]) - analytic_slip(y)));
            sum_drho += static_cast<long double>(cur.drho[i]);
        }

    // CSV: column x = 0 (the flow is uniform in x up to round-off).
    std::filesystem::create_directories("output");
    char name[256];
    std::snprintf(name, sizeof name, "output/poiseuille_H%d_nu%.6g_lambda%.6g.csv", H, nu, lambda);
    if (std::FILE* fp = std::fopen(name, "w")) {
        std::fprintf(fp, "y,u_numeric,u_analytic,abs_error\n");
        for (int y = 1; y < ny - 1; ++y) {
            const double un = static_cast<double>(cur.ux[static_cast<std::size_t>(g.node(0, y))]);
            const double ua = analytic(y);
            std::fprintf(fp, "%d,%.17g,%.17g,%.17g\n", y, un, ua, std::abs(un - ua));
        }
        std::fclose(fp);
    }

    Result r;
    r.H = H;
    r.nu = nu;
    r.lambda = lambda;
    r.steps = steps;
    r.change = change;
    r.l2 = std::sqrt(num / den);
    r.linf = linf / umax_a;
    r.l2_stored = std::sqrt(num_s / den);
    r.max_uy = max_uy;
    r.max_drho = max_drho;
    r.slip = static_cast<double>(du / kUmax);
    r.err_slip = static_cast<double>(err_slip / kUmax);
    r.mean_drho = static_cast<double>(sum_drho / static_cast<long double>(kNx * H));
    r.capped = capped;
    return r;
}

template <typename Storage, typename Compute>
void run_all() {
    const int widths[] = {8, 16, 32, 64};
    const double nus[] = {0.01, 1.0 / 6.0, 0.5};
    const double lambdas[] = {3.0 / 16.0, 0.25};

    for (const double nu : nus)
        for (const double lambda : lambdas) {
            std::vector<Result> rs;
            for (const int H : widths) {
                const Result r = run_case<Storage, Compute>(H, nu, lambda);
                rs.push_back(r);
                std::printf("H=%d nu=%.6g lambda=%.6g steps=%ld final_change=%.3e%s l2=%.6e linf=%.6e "
                            "max_uy=%.3e max_drho=%.3e l2_stored=%.6e slip=%.6e err_slip=%.6e "
                            "mean_drho=%.3e\n",
                            r.H, r.nu, r.lambda, r.steps, r.change,
                            r.capped ? " (CAP REACHED)" : "", r.l2, r.linf, r.max_uy, r.max_drho,
                            r.l2_stored, r.slip, r.err_slip, r.mean_drho);
                std::fflush(stdout);
            }
            for (std::size_t i = 1; i < rs.size(); ++i)
                std::printf("order nu=%.6g lambda=%.6g H=%d->%d: l2 %.3f linf %.3f\n", nu, lambda,
                            rs[i - 1].H, rs[i].H, std::log2(rs[i - 1].l2 / rs[i].l2),
                            std::log2(rs[i - 1].linf / rs[i].linf));
        }
}

[[noreturn]] void usage_error(const std::string& msg) {
    std::fprintf(stderr, "lbm_poiseuille: %s\n", msg.c_str());
    std::fprintf(stderr, "usage: lbm_poiseuille [--compute double|float] [--storage same|float|p16]\n"
                         "  supported combinations: double/same, double/float, float/same,\n"
                         "  float/float (= float/same), float/p16\n");
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    std::string compute = "double", storage = "same";
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        std::string value;
        const std::size_t eq = key.find('=');
        if (eq != std::string::npos) {
            value = key.substr(eq + 1);
            key = key.substr(0, eq);
        } else if (key == "--help" || key == "-h") {
            usage_error("help");
        } else {
            if (i + 1 >= argc) usage_error("missing value for " + key);
            value = argv[++i];
        }
        if (key == "--compute")
            compute = value;
        else if (key == "--storage")
            storage = value;
        else
            usage_error("unknown option " + key);
    }
    if (compute != "double" && compute != "float") usage_error("invalid compute type '" + compute + "'");
    if (storage != "same" && storage != "float" && storage != "p16")
        usage_error("invalid storage type '" + storage + "'");
    if (storage == "same") storage = compute;
    if (compute == "double" && storage == "p16")
        usage_error("p16 storage requires --compute float (no StorageConversion<P16, double>)");

    try {
        const auto t0 = std::chrono::steady_clock::now();
        std::printf("compute %s, storage %s\n", compute.c_str(), storage.c_str());
        if (compute == "double" && storage == "double")
            run_all<double, double>();
        else if (compute == "double")
            run_all<float, double>();
        else if (storage == "float")
            run_all<float, float>();
        else
            run_all<lbm::P16, float>();
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("total run time: %.1f s\n", secs);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lbm_poiseuille: %s\n", e.what());
        return 1;
    }
}
