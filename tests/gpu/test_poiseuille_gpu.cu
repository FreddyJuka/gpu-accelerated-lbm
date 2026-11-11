// Poiseuille validation on the GPU: same cases, checks and bounds as
// test_poiseuille.cpp (see poiseuille_common.hpp). This target keeps nvcc's
// default FMA contraction (production flags); the tolerance derivation holds
// for fused and unfused arithmetic alike.

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "../poiseuille_common.hpp"
#include "lbm/cuda/check.hpp"
#include "lbm/cuda/device_buffer.hpp"
#include "lbm/cuda/step.cuh"

namespace {

using namespace poiseuille;

template <typename T>
class GpuStepper {
public:
    explicit GpuStepper(const ChannelSetup<T>& s)
        : s_(s),
          n_(L::Q * s.g.num_nodes()),
          da_(n_),
          db_(n_),
          d_flags_(s.g.num_nodes()),
          host_(static_cast<std::size_t>(n_), T(0)) {
        da_.copy_from_host(host_.data(), n_);
        db_.copy_from_host(host_.data(), n_);
        d_flags_.copy_from_host(s.flags.data(), s.g.num_nodes());
    }

    void advance(int n) {
        for (int i = 0; i < n; ++i) {
            lbm::step_gpu<L, T>(s_.g, d_flags_.data(), da_.data(), db_.data(), s_.coll);
            std::swap(da_, db_);
        }
        LBM_CUDA_CHECK(cudaDeviceSynchronize());
        da_.copy_to_host(host_.data(), n_);
    }
    const std::vector<T>& state() const { return host_; }

private:
    const ChannelSetup<T>& s_;
    int n_;
    lbm::DeviceBuffer<T> da_, db_;
    lbm::DeviceBuffer<lbm::NodeType> d_flags_;
    std::vector<T> host_;
};

template <typename T>
void run_case(const Case& c) {
    const ChannelSetup<T> s(c);
    GpuStepper<T> stepper(s);
    check_case(s, stepper);
}

auto param_name = [](const ::testing::TestParamInfo<Case>& info) { return case_name(info.param); };

class PoiseuilleGpu : public ::testing::TestWithParam<Case> {};

TEST_P(PoiseuilleGpu, MatchesShiftedParabola) { run_case<double>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Channel, PoiseuilleGpu, ::testing::ValuesIn(all_cases()), param_name);

// Single precision: the cases of float_cases(), see "Single precision" in
// poiseuille_common.hpp.
class PoiseuilleGpuFloat : public ::testing::TestWithParam<Case> {};

TEST_P(PoiseuilleGpuFloat, MatchesShiftedParabola) { run_case<float>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Channel, PoiseuilleGpuFloat, ::testing::ValuesIn(float_cases()),
                         param_name);

}  // namespace
