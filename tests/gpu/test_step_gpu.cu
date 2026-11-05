// Exact CPU-vs-GPU comparison of the periodic pull stream-collide step.
//
// The GPU kernel and the CPU reference call the same node function
// (pull_stream_collide_periodic), so with identical IEEE operations in the
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

using L = lbm::D2Q9;
using Grid = lbm::Grid2D<L>;

namespace {

struct GridSize {
    int nx, ny;
};
// 7 x 5: single partial block. 200 x 37: two blocks along x, the second one
// partial (72 of 128 threads active).
const GridSize kSizes[] = {{7, 5}, {200, 37}};

// Non-uniform shifted populations: equilibrium of a smooth drho/u field plus
// the deterministic perturbation p_q = 1e-3 * w_q * m_q, m_q = ((3q+1) mod 7) - 3.
template <typename T>
std::vector<T> nonuniform_field(const Grid& g) {
    std::vector<T> f(static_cast<std::size_t>(L::Q * g.num_nodes()));
    for (int y = 0; y < g.ny(); ++y)
        for (int x = 0; x < g.nx(); ++x) {
            const T drho = T(0.01) * static_cast<T>(x % 11) - T(0.005) * static_cast<T>(y % 7);
            const T u[2] = {T(0.02) + T(0.004) * static_cast<T>(y % 5),
                            T(-0.01) + T(0.003) * static_cast<T>(x % 9)};
            for (int q = 0; q < L::Q; ++q) {
                const T p = T(1e-3) * L::w<T>(q) * static_cast<T>(((3 * q + 1) % 7) - 3);
                f[static_cast<std::size_t>(g.idx(q, x, y))] =
                    lbm::equilibrium_shifted<L, T>(q, drho, u) + p;
            }
        }
    return f;
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

// a. One step: GPU equals CPU bitwise.
TYPED_TEST(StepGpu, OneStepMatchesCpu) {
    using T = TypeParam;
    for (const GridSize s : kSizes) {
        const Grid g(s.nx, s.ny);
        const auto trt = lbm::TRT<T>::template from_viscosity<L>(0.01, 3.0 / 16.0);
        const std::vector<T> init = nonuniform_field<T>(g);
        const int n = static_cast<int>(init.size());

        std::vector<T> cpu(init.size(), T(-1));
        lbm::step_periodic_cpu<L, T>(g, init.data(), cpu.data(), trt);

        lbm::DeviceBuffer<T> d_src(n), d_dst(n);
        d_src.copy_from_host(init.data(), n);
        const std::vector<T> poison(init.size(), T(-1));
        d_dst.copy_from_host(poison.data(), n);
        lbm::step_periodic_gpu<L, T>(g, d_src.data(), d_dst.data(), trt);
        LBM_CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<T> gpu(init.size());
        d_dst.copy_to_host(gpu.data(), n);

        expect_equal(gpu, cpu, std::to_string(s.nx) + "x" + std::to_string(s.ny));
    }
}

// b. 100 steps with alternating buffers on both sides; compared after step 1,
// 50 and 100.
TYPED_TEST(StepGpu, HundredStepsMatchCpu) {
    using T = TypeParam;
    for (const GridSize s : kSizes) {
        const Grid g(s.nx, s.ny);
        const auto trt = lbm::TRT<T>::template from_viscosity<L>(0.01, 3.0 / 16.0);
        const std::vector<T> init = nonuniform_field<T>(g);
        const int n = static_cast<int>(init.size());

        std::vector<T> ca = init, cb(init.size(), T(-1));
        lbm::DeviceBuffer<T> da(n), db(n);
        da.copy_from_host(init.data(), n);
        db.copy_from_host(cb.data(), n);

        std::vector<T> gpu(init.size());
        for (int step = 1; step <= 100; ++step) {
            lbm::step_periodic_cpu<L, T>(g, ca.data(), cb.data(), trt);
            std::swap(ca, cb);
            lbm::step_periodic_gpu<L, T>(g, da.data(), db.data(), trt);
            std::swap(da, db);

            if (step == 1 || step == 50 || step == 100) {
                LBM_CUDA_CHECK(cudaDeviceSynchronize());
                da.copy_to_host(gpu.data(), n);
                expect_equal(gpu, ca,
                             std::to_string(s.nx) + "x" + std::to_string(s.ny) + " step " +
                                 std::to_string(step));
            }
        }
    }
}

// c. Launcher rejection.
TYPED_TEST(StepGpu, LauncherRejectsInvalidArguments) {
    using T = TypeParam;
    const Grid g(7, 5);
    const auto trt = lbm::TRT<T>::template from_viscosity<L>(0.01, 3.0 / 16.0);
    lbm::DeviceBuffer<T> a(L::Q * g.num_nodes()), b(L::Q * g.num_nodes());
    auto step = [&](const T* src, T* dst) { lbm::step_periodic_gpu<L, T>(g, src, dst, trt); };

    EXPECT_THROW(step(nullptr, b.data()), std::invalid_argument);
    EXPECT_THROW(step(a.data(), nullptr), std::invalid_argument);
    EXPECT_THROW(step(nullptr, nullptr), std::invalid_argument);
    EXPECT_THROW(step(a.data(), a.data()), std::invalid_argument);
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

    try {
        lbm::step_periodic_gpu<L, T>(g, a.data(), b.data(), trt);
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
