// Exact CPU-vs-GPU comparison of the pull stream-collide step with bounce-back.
//
// The GPU kernel and the CPU reference call the same node function
// (pull_stream_collide), so with identical IEEE operations in the
// same order the results are bitwise equal. This target is compiled with
// --fmad=false to rule out fused multiply-add contraction on the device (see
// tests/CMakeLists.txt). All comparisons are EXPECT_EQ.

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lbm/collision/trt.hpp"
#include "lbm/cpu/step.hpp"
#include "lbm/cuda/check.hpp"
#include "lbm/cuda/device_buffer.hpp"
#include "lbm/cuda/step.cuh"
#include "lbm/equilibrium.hpp"
#include "lbm/grid.hpp"
#include "lbm/lattice/d2q9.hpp"
#include "support/fields.hpp"
#include "support/fp_environment.hpp"

using namespace lbmtest;

namespace {

struct GridSize {
    int nx, ny;
};
// 7 x 5: single partial block. 200 x 37: two blocks along x, the second one
// partial (72 of 128 threads active).
const GridSize kSizes[] = {{7, 5}, {200, 37}};

std::string label(const GridSize& s, Geometry geo) {
    return std::to_string(s.nx) + "x" + std::to_string(s.ny) +
           (geo == Geometry::Channel ? " G1" : " G2");
}

template <typename T>
void expect_equal(const std::vector<T>& a, const std::vector<T>& b, const std::string& what) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            ADD_FAILURE() << what << ": first mismatch at index " << i << ": " << a[i] << " vs " << b[i];
            return;
        }
}

template <typename T>
class StepGpu : public ::testing::Test {};
using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(StepGpu, PrecisionTypes);

} // namespace

// Bodies shared by the unforced and the forced collision. Trt is
// TRT<T> or TRT<T, UniformForce<T, 2>>.
namespace {

template <typename T, typename Trt>
void check_one_step(const Trt& trt, const std::string& tag) {
    for (const GridSize s : kSizes)
        for (const Geometry geo : kGeometries) {
            const Grid g(s.nx, s.ny);
            const std::vector<T> init = nonuniform_field<T>(g);
            const std::vector<lbm::NodeType> flags = make_flags(g, geo);
            const int n = static_cast<int>(init.size());

            std::vector<T> cpu = init;
            lbm::step_cpu<L, T>(g, flags.data(), init.data(), cpu.data(), trt);

            lbm::DeviceBuffer<T> d_src(n), d_dst(n);
            lbm::DeviceBuffer<lbm::NodeType> d_flags(g.num_nodes());
            d_flags.copy_from_host(flags.data(), g.num_nodes());
            d_src.copy_from_host(init.data(), n);
            d_dst.copy_from_host(init.data(), n);
            lbm::step_gpu<L, T>(g, d_flags.data(), d_src.data(), d_dst.data(), trt);
            LBM_CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<T> gpu(init.size());
            d_dst.copy_to_host(gpu.data(), n);

            expect_equal(gpu, cpu, tag + label(s, geo));
        }
}

template <typename T, typename Trt>
void check_hundred_steps(const Trt& trt, const std::string& tag) {
    for (const GridSize s : kSizes)
        for (const Geometry geo : kGeometries) {
            const Grid g(s.nx, s.ny);
            const std::vector<T> init = nonuniform_field<T>(g);
            const std::vector<lbm::NodeType> flags = make_flags(g, geo);
            const int n = static_cast<int>(init.size());

            std::vector<T> ca = init, cb = init;
            lbm::DeviceBuffer<T> da(n), db(n);
            lbm::DeviceBuffer<lbm::NodeType> d_flags(g.num_nodes());
            d_flags.copy_from_host(flags.data(), g.num_nodes());
            da.copy_from_host(init.data(), n);
            db.copy_from_host(init.data(), n);

            std::vector<T> gpu(init.size());
            for (int step = 1; step <= 100; ++step) {
                lbm::step_cpu<L, T>(g, flags.data(), ca.data(), cb.data(), trt);
                std::swap(ca, cb);
                lbm::step_gpu<L, T>(g, d_flags.data(), da.data(), db.data(), trt);
                std::swap(da, db);

                if (step == 1 || step == 50 || step == 100) {
                    LBM_CUDA_CHECK(cudaDeviceSynchronize());
                    da.copy_to_host(gpu.data(), n);
                    expect_equal(gpu, ca, tag + label(s, geo) + " step " + std::to_string(step));
                }
            }
        }
}

// Both components nonzero, so the symmetric and the antisymmetric source are
// both active.
template <typename T>
lbm::TRT<T, lbm::UniformForce<T, 2>> forced_trt() {
    const double F[2] = {3e-5, -7e-5};
    return lbm::TRT<T, lbm::UniformForce<T, 2>>::template from_viscosity<L>(
        0.01, 3.0 / 16.0, lbm::UniformForce<T, 2>::create(F));
}

} // namespace

// a. One step: GPU equals CPU bitwise, with walls (G1, G2). Both dst buffers
// start as a copy of the source so that the untouched wall-node values compare
// equal.
TYPED_TEST(StepGpu, OneStepMatchesCpu) {
    check_one_step<TypeParam>(lbm::TRT<TypeParam>::template from_viscosity<L>(0.01, 3.0 / 16.0),
                              "unforced ");
}

// a2. The same with the forced collision (Guo source).
TYPED_TEST(StepGpu, OneStepForcedMatchesCpu) {
    check_one_step<TypeParam>(forced_trt<TypeParam>(), "forced ");
}

// b. 100 steps with alternating buffers on both sides; compared after step 1,
// 50 and 100. Both buffer pairs start as copies of the initial field.
TYPED_TEST(StepGpu, HundredStepsMatchCpu) {
    check_hundred_steps<TypeParam>(lbm::TRT<TypeParam>::template from_viscosity<L>(0.01, 3.0 / 16.0),
                                   "unforced ");
}

// b2. The same with the forced collision.
TYPED_TEST(StepGpu, HundredStepsForcedMatchesCpu) {
    check_hundred_steps<TypeParam>(forced_trt<TypeParam>(), "forced ");
}

// c. Launcher rejection.
TYPED_TEST(StepGpu, LauncherRejectsInvalidArguments) {
    using T = TypeParam;
    const Grid g(7, 5);
    const auto trt = lbm::TRT<T>::template from_viscosity<L>(0.01, 3.0 / 16.0);
    lbm::DeviceBuffer<T> a(L::Q * g.num_nodes()), b(L::Q * g.num_nodes());
    lbm::DeviceBuffer<lbm::NodeType> flags(g.num_nodes());
    const lbm::NodeType* f = flags.data();
    auto step = [&](const lbm::NodeType* fl, const T* src, T* dst) {
        lbm::step_gpu<L, T>(g, fl, src, dst, trt);
    };

    EXPECT_THROW(step(f, nullptr, b.data()), std::invalid_argument);
    EXPECT_THROW(step(f, a.data(), nullptr), std::invalid_argument);
    EXPECT_THROW(step(f, nullptr, nullptr), std::invalid_argument);
    EXPECT_THROW(step(f, a.data(), a.data()), std::invalid_argument);
    EXPECT_THROW(step(nullptr, a.data(), b.data()), std::invalid_argument);
}

// c2. ny above the device's grid y limit is a valid Grid2D but is rejected by
// the launcher before anything is launched (the buffers are never accessed).
TYPED_TEST(StepGpu, LauncherRejectsNyAboveGridLimit) {
    using T = TypeParam;
    int device = 0;
    LBM_CUDA_CHECK(cudaGetDevice(&device));
    int max_grid_y = 0;
    LBM_CUDA_CHECK(cudaDeviceGetAttribute(&max_grid_y, cudaDevAttrMaxGridDimY, device));
    ASSERT_GT(max_grid_y, 0);

    const int ny = max_grid_y + 1;
    ASSERT_NO_THROW(Grid(1, ny));
    const Grid g(1, ny);
    const auto trt = lbm::TRT<T>::template from_viscosity<L>(0.01, 3.0 / 16.0);
    lbm::DeviceBuffer<T> a(1), b(1);
    lbm::DeviceBuffer<lbm::NodeType> flags(1);

    try {
        lbm::step_gpu<L, T>(g, flags.data(), a.data(), b.data(), trt);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("ny"), std::string::npos) << msg;
        EXPECT_NE(msg.find(std::to_string(ny)), std::string::npos) << msg;
    }
}

// d. DeviceBuffer.
TYPED_TEST(StepGpu, DeviceBufferRoundTripAndMove) {
    using T = TypeParam;
    const int n = 1000;
    std::vector<T> host(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) host[static_cast<std::size_t>(i)] = static_cast<T>(i) * T(0.37) - T(11);

    lbm::DeviceBuffer<T> buf(n);
    EXPECT_EQ(buf.size(), n);
    EXPECT_NE(buf.data(), nullptr);
    buf.copy_from_host(host.data(), n);
    std::vector<T> back(static_cast<std::size_t>(n), T(0));
    buf.copy_to_host(back.data(), n);
    EXPECT_EQ(back, host);

    // Size mismatch.
    EXPECT_THROW(buf.copy_from_host(host.data(), n - 1), std::invalid_argument);
    EXPECT_THROW(buf.copy_to_host(back.data(), n + 1), std::invalid_argument);

    // Move construction: source is left empty, contents are preserved.
    const T* old_ptr = buf.data();
    lbm::DeviceBuffer<T> moved(std::move(buf));
    EXPECT_EQ(buf.data(), nullptr);  // NOLINT: moved-from state is specified
    EXPECT_EQ(buf.size(), 0);
    EXPECT_EQ(moved.data(), old_ptr);
    EXPECT_EQ(moved.size(), n);
    std::vector<T> back2(static_cast<std::size_t>(n), T(0));
    moved.copy_to_host(back2.data(), n);
    EXPECT_EQ(back2, host);

    // Move assignment.
    lbm::DeviceBuffer<T> assigned(5);
    assigned = std::move(moved);
    EXPECT_EQ(moved.data(), nullptr);  // NOLINT
    EXPECT_EQ(moved.size(), 0);
    EXPECT_EQ(assigned.size(), n);
    EXPECT_EQ(assigned.data(), old_ptr);

    // Null host pointers are rejected before any CUDA call.
    EXPECT_THROW(assigned.copy_from_host(nullptr, n), std::invalid_argument);
    EXPECT_THROW(assigned.copy_to_host(nullptr, n), std::invalid_argument);

    // Invalid counts.
    EXPECT_THROW(lbm::DeviceBuffer<T>(0), std::invalid_argument);
    EXPECT_THROW(lbm::DeviceBuffer<T>(-1), std::invalid_argument);
}

// e. CUDA error checking.
TEST(CudaCheck, ThrowsWithExpressionText) {
    try {
        LBM_CUDA_CHECK(cudaSetDevice(9999));
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("cudaSetDevice(9999)"), std::string::npos) << e.what();
    }
    // Clear the error so later tests are unaffected.
    (void)cudaGetLastError();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
