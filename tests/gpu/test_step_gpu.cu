// Exact CPU-vs-GPU comparison of the pull stream-collide step with bounce-back.
//
// The GPU kernel and the CPU reference call the same node function
// (pull_stream_collide), so with identical IEEE operations in the
// same order the results are bitwise equal. This target is compiled with
// --fmad=false to rule out fused multiply-add contraction on the device (see
// tests/CMakeLists.txt). All comparisons are EXPECT_EQ.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
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
#include "lbm/storage.hpp"
#include "support/fields.hpp"
#include "support/fp_environment.hpp"

using namespace lbmtest;

namespace {

struct GridSize {
    int nx, ny;
};
// 7 x 5 = 35 nodes: a single partial block (35 of 128 threads active).
// 200 x 37 = 7400 nodes: 57 full blocks of 128 (7296 nodes) plus one partial
// block of 104 threads; blocks straddle row boundaries.
const GridSize kSizes[] = {{7, 5}, {200, 37}};

std::string label(const GridSize& s, Geometry geo) {
    return std::to_string(s.nx) + "x" + std::to_string(s.ny) +
           (geo == Geometry::Channel ? " G1" : " G2");
}

template <typename T>
void expect_equal(const std::vector<T>& a, const std::vector<T>& b, const std::string& what) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        if (raw(a[i]) != raw(b[i])) {
            ADD_FAILURE() << what << ": first mismatch at index " << i << ": " << raw(a[i]) << " vs "
                          << raw(b[i]);
            return;
        }
}

template <typename T>
class StepGpu : public ::testing::Test {};
using PrecisionTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(StepGpu, PrecisionTypes);

} // namespace

// Bodies shared by the unforced and the forced collision. Trt is
// TRT<T> or TRT<T, UniformForce<T, 2>>; for the mixed-precision tests T is
// the storage type (float, or lbm::P16) and Trt computes in double (or float).
namespace {

template <typename T, typename Trt>
void check_one_step(const Trt& trt, const std::string& tag) {
    for (const GridSize s : kSizes)
        for (const Geometry geo : kGeometries) {
            const Grid g(s.nx, s.ny);
            const std::vector<T> init = nonuniform_stored_field<T>(g);
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
            const std::vector<T> init = nonuniform_stored_field<T>(g);
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

// b3. Mixed precision: float storage (host vectors and device buffers of
// float) with double compute (TRT<double>). The shared bodies take the storage
// type as T and the collision as given, so the comparison is the same as
// above: bitwise, 7 x 5 and 200 x 37, channel and single wall. Load (float ->
// double, exact) and store (double -> float, one IEEE rounding to nearest) are
// single operations on both sides.
TEST(StepGpuMixed, OneStepMatchesCpu) {
    check_one_step<float>(lbm::TRT<double>::from_viscosity<L>(0.01, 3.0 / 16.0),
                          "mixed unforced ");
}

TEST(StepGpuMixed, OneStepForcedMatchesCpu) {
    check_one_step<float>(forced_trt<double>(), "mixed forced ");
}

TEST(StepGpuMixed, HundredStepsMatchCpu) {
    check_hundred_steps<float>(lbm::TRT<double>::from_viscosity<L>(0.01, 3.0 / 16.0),
                               "mixed unforced ");
}

TEST(StepGpuMixed, HundredStepsForcedMatchesCpu) {
    check_hundred_steps<float>(forced_trt<double>(), "mixed forced ");
}

// b4. 16-bit storage: buffers of lbm::P16 (host vectors and device buffers)
// with float compute. Same bodies and comparison: bitwise on the 16-bit
// patterns, 7 x 5 and 200 x 37, channel and single wall, 1 step and 100 steps
// (compared after step 1, 50 and 100), unforced and forced. Load and store are
// integer operations on the bit patterns (lbm/storage.hpp), identical on both
// sides by construction; the float collision between them is the one of the
// float tests above. The initial field is the float field converted with
// store. Over 100 steps the stored rounding of every step feeds the next one,
// so a single differing conversion on either side would show.
TEST(StepGpuP16, OneStepMatchesCpu) {
    check_one_step<lbm::P16>(lbm::TRT<float>::from_viscosity<L>(0.01, 3.0 / 16.0), "P16 unforced ");
}

TEST(StepGpuP16, OneStepForcedMatchesCpu) {
    check_one_step<lbm::P16>(forced_trt<float>(), "P16 forced ");
}

TEST(StepGpuP16, HundredStepsMatchCpu) {
    check_hundred_steps<lbm::P16>(lbm::TRT<float>::from_viscosity<L>(0.01, 3.0 / 16.0),
                                  "P16 unforced ");
}

TEST(StepGpuP16, HundredStepsForcedMatchesCpu) {
    check_hundred_steps<lbm::P16>(forced_trt<float>(), "P16 forced ");
}

// b5. The conversion alone on the device: store of 2^20 binary32 values spread
// over all bit patterns (every 4093rd, a prime stride; zeros, subnormals, the
// overflow range, infinities and NaNs included) and load of all 65536
// patterns, each compared bitwise with the host result of the same function.
// The host functions are tested against the exact reference in
// test_storage.cpp.
namespace {

__global__ void p16_store_kernel(const float* in, lbm::P16* out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = lbm::StorageConversion<lbm::P16, float>::store(in[i]);
}

__global__ void p16_load_kernel(const lbm::P16* in, float* out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = lbm::StorageConversion<lbm::P16, float>::load(in[i]);
}

std::uint32_t bits_of(float x) {
    std::uint32_t b;
    std::memcpy(&b, &x, sizeof b);
    return b;
}

} // namespace

TEST(StepGpuP16, ConversionMatchesHostBitwise) {
    using C16 = lbm::StorageConversion<lbm::P16, float>;
    const int n_store = 1 << 20;
    std::vector<float> x(static_cast<std::size_t>(n_store));
    for (int i = 0; i < n_store; ++i) {
        const std::uint32_t b = static_cast<std::uint32_t>(i) * 4093u;
        std::memcpy(&x[static_cast<std::size_t>(i)], &b, sizeof b);
    }
    lbm::DeviceBuffer<float> d_x(n_store);
    lbm::DeviceBuffer<lbm::P16> d_s(n_store);
    d_x.copy_from_host(x.data(), n_store);
    p16_store_kernel<<<(n_store + 255) / 256, 256>>>(d_x.data(), d_s.data(), n_store);
    LBM_CUDA_CHECK(cudaGetLastError());
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<lbm::P16> stored(static_cast<std::size_t>(n_store));
    d_s.copy_to_host(stored.data(), n_store);
    int mismatches = 0;
    for (int i = 0; i < n_store; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        // A NaN input is not guaranteed to keep its sign bit on its way through
        // host code; compare the magnitude bits there.
        const unsigned mask = std::isnan(x[k]) ? 0x7FFFu : 0xFFFFu;
        if ((stored[k].bits & mask) != (C16::store(x[k]).bits & mask) && ++mismatches <= 10)
            ADD_FAILURE() << "store, input bits " << bits_of(x[k]) << ": device " << stored[k].bits
                          << ", host " << C16::store(x[k]).bits;
    }
    EXPECT_EQ(mismatches, 0);

    const int n_load = 1 << 16;
    std::vector<lbm::P16> patterns(static_cast<std::size_t>(n_load));
    for (int i = 0; i < n_load; ++i) patterns[static_cast<std::size_t>(i)].bits = static_cast<std::uint16_t>(i);
    lbm::DeviceBuffer<lbm::P16> d_p(n_load);
    lbm::DeviceBuffer<float> d_v(n_load);
    d_p.copy_from_host(patterns.data(), n_load);
    p16_load_kernel<<<(n_load + 255) / 256, 256>>>(d_p.data(), d_v.data(), n_load);
    LBM_CUDA_CHECK(cudaGetLastError());
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> loaded(static_cast<std::size_t>(n_load));
    d_v.copy_to_host(loaded.data(), n_load);
    for (int i = 0; i < n_load; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        ASSERT_EQ(bits_of(loaded[k]), bits_of(C16::load(patterns[k]))) << "load, pattern " << i;
    }
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

// c2. Tall, narrow grids with ny above the maximum grid y dimension (65535)
// of the former two-dimensional launch run on the GPU and match the CPU
// bitwise after a few steps.
TYPED_TEST(StepGpu, TallNarrowGridsMatchCpu) {
    using T = TypeParam;
    const auto trt = lbm::TRT<T>::template from_viscosity<L>(0.01, 3.0 / 16.0);
    for (const GridSize s : {GridSize{1, 70000}, GridSize{3, 70000}})
        for (const Geometry geo : kGeometries) {
            ASSERT_GT(s.ny, 65535);
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

            for (int step = 0; step < 5; ++step) {
                lbm::step_cpu<L, T>(g, flags.data(), ca.data(), cb.data(), trt);
                std::swap(ca, cb);
                lbm::step_gpu<L, T>(g, d_flags.data(), da.data(), db.data(), trt);
                std::swap(da, db);
            }
            LBM_CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<T> gpu(init.size());
            da.copy_to_host(gpu.data(), n);
            expect_equal(gpu, ca, label(s, geo) + " 5 steps");
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
