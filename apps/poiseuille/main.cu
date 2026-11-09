// Exploratory measurement: steady body-force-driven channel flow (Poiseuille)
// against the analytical profile, for several widths, viscosities and magic
// parameters. Double precision on the GPU, TRT with Guo forcing, halfway
// bounce-back walls at y = 0 and y = ny - 1.
//
// Velocity measurement. The stored populations are post-collision. The Guo
// half-force velocity u = (j + F/2) / rho is defined on the collision input,
// i.e. the populations after the gather (streaming + bounce-back). The
// velocity is therefore obtained by running the gather of the production node
// function (pull_stream_collide with an identity collision) on the stored
// populations and applying moments_shifted_forced to the result. As a
// diagnostic, the relative L2 error of moments_shifted_forced applied directly
// to the stored post-collision populations is also printed ("l2_stored").

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
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

namespace {

using L = lbm::D2Q9;
using Grid = lbm::Grid2D<L>;
using Force = lbm::UniformForce<double, 2>;
using Collision = lbm::TRT<double, Force>;

constexpr double kUmax = 0.01;
constexpr int kCheckEvery = 1000;
constexpr long kStepCap = 5000000;
constexpr double kChangeTol = 1e-14 * kUmax;
constexpr int kNx = 4;

struct IdentityCollision {
    template <typename Lat, typename T>
    LBM_HD void collide(T (&)[Lat::Q]) const {}
};

struct Fields {
    std::vector<double> ux, uy, drho;  // per node, index y * nx + x
};

// Macroscopic fields of the stored (post-collision) populations. If gather is
// true, the gather is applied first (see the file comment).
Fields fields_from(const Grid& g, const std::vector<lbm::NodeType>& flags,
                   const std::vector<double>& stored, const Force& force, bool gather) {
    std::vector<double> work = stored;
    if (gather) {
        for (int y = 0; y < g.ny(); ++y)
            for (int x = 0; x < g.nx(); ++x)
                lbm::pull_stream_collide<L, double>(g, flags.data(), stored.data(), work.data(),
                                                    IdentityCollision{}, x, y);
    }
    Fields r;
    const std::size_t n = static_cast<std::size_t>(g.num_nodes());
    r.ux.assign(n, 0.0);
    r.uy.assign(n, 0.0);
    r.drho.assign(n, 0.0);
    const double F[2] = {force.F(0), force.F(1)};
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x) {
            if (flags[static_cast<std::size_t>(g.node(x, y))] == lbm::NodeType::Wall) continue;
            double f[L::Q], u[2], drho;
            for (int q = 0; q < L::Q; ++q) f[q] = work[static_cast<std::size_t>(g.idx(q, x, y))];
            lbm::moments_shifted_forced<L, double>(f, F, drho, u);
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
    bool capped;
};

Result run_case(int H, double nu, double lambda) {
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
    const Collision coll = Collision::from_viscosity<L>(nu, lambda, force);

    const std::vector<double> zeros(static_cast<std::size_t>(n), 0.0);
    lbm::DeviceBuffer<double> da(n), db(n);
    lbm::DeviceBuffer<lbm::NodeType> d_flags(g.num_nodes());
    da.copy_from_host(zeros.data(), n);
    db.copy_from_host(zeros.data(), n);
    d_flags.copy_from_host(flags.data(), g.num_nodes());

    std::vector<double> host(static_cast<std::size_t>(n));
    std::vector<double> ux_prev(static_cast<std::size_t>(g.num_nodes()), 0.0);  // rest

    long steps = 0;
    double change = 0.0;
    bool capped = false;
    Fields cur;
    for (;;) {
        for (int k = 0; k < kCheckEvery; ++k) {
            lbm::step_gpu<L, double>(g, d_flags.data(), da.data(), db.data(), coll);
            std::swap(da, db);
        }
        steps += kCheckEvery;
        LBM_CUDA_CHECK(cudaDeviceSynchronize());
        da.copy_to_host(host.data(), n);
        cur = fields_from(g, flags, host, force, true);
        change = 0.0;
        for (std::size_t i = 0; i < cur.ux.size(); ++i)
            if (flags[i] == lbm::NodeType::Fluid) change = std::max(change, std::abs(cur.ux[i] - ux_prev[i]));
        ux_prev = cur.ux;
        if (change <= kChangeTol) break;
        if (steps >= kStepCap) {
            capped = true;
            break;
        }
    }

    const Fields stored = fields_from(g, flags, host, force, false);

    // Errors against u_a(y) = Fx / (2 nu) (y - 0.5)(ny - 1.5 - y).
    auto analytic = [&](int y) { return Fx / (2.0 * nu) * (y - 0.5) * (ny - 1.5 - y); };
    double num = 0.0, den = 0.0, num_s = 0.0, linf = 0.0, umax_a = 0.0, max_uy = 0.0, max_drho = 0.0;
    for (int y = 1; y < ny - 1; ++y)
        for (int x = 0; x < kNx; ++x) {
            const std::size_t i = static_cast<std::size_t>(g.node(x, y));
            const double ua = analytic(y);
            const double e = cur.ux[i] - ua;
            num += e * e;
            den += ua * ua;
            const double es = stored.ux[i] - ua;
            num_s += es * es;
            linf = std::max(linf, std::abs(e));
            umax_a = std::max(umax_a, ua);
            max_uy = std::max(max_uy, std::abs(cur.uy[i]));
            max_drho = std::max(max_drho, std::abs(cur.drho[i]));
        }

    // CSV: column x = 0 (the flow is uniform in x up to round-off).
    std::filesystem::create_directories("output");
    char name[256];
    std::snprintf(name, sizeof name, "output/poiseuille_H%d_nu%.6g_lambda%.6g.csv", H, nu, lambda);
    if (std::FILE* fp = std::fopen(name, "w")) {
        std::fprintf(fp, "y,u_numeric,u_analytic,abs_error\n");
        for (int y = 1; y < ny - 1; ++y) {
            const double un = cur.ux[static_cast<std::size_t>(g.node(0, y))];
            const double ua = analytic(y);
            std::fprintf(fp, "%d,%.17g,%.17g,%.17g\n", y, un, ua, std::abs(un - ua));
        }
        std::fclose(fp);
    }

    return {H, nu, lambda, steps, change, std::sqrt(num / den), linf / umax_a,
            std::sqrt(num_s / den), max_uy, max_drho, capped};
}

}  // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    const int widths[] = {8, 16, 32, 64};
    const double nus[] = {0.01, 1.0 / 6.0, 0.5};
    const double lambdas[] = {3.0 / 16.0, 0.25};

    for (const double nu : nus)
        for (const double lambda : lambdas) {
            std::vector<Result> rs;
            for (const int H : widths) {
                const Result r = run_case(H, nu, lambda);
                rs.push_back(r);
                std::printf("H=%d nu=%.6g lambda=%.6g steps=%ld final_change=%.3e%s l2=%.6e linf=%.6e "
                            "max_uy=%.3e max_drho=%.3e l2_stored=%.6e\n",
                            r.H, r.nu, r.lambda, r.steps, r.change,
                            r.capped ? " (CAP REACHED)" : "", r.l2, r.linf, r.max_uy, r.max_drho,
                            r.l2_stored);
                std::fflush(stdout);
            }
            for (std::size_t i = 1; i < rs.size(); ++i)
                std::printf("order nu=%.6g lambda=%.6g H=%d->%d: l2 %.3f linf %.3f\n", nu, lambda,
                            rs[i - 1].H, rs[i].H, std::log2(rs[i - 1].l2 / rs[i].l2),
                            std::log2(rs[i - 1].linf / rs[i].linf));
        }

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("total run time: %.1f s\n", secs);
    return 0;
}
