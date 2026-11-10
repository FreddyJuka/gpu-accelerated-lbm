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

class GpuStepper {
public:
    explicit GpuStepper(const ChannelSetup& s)
        : s_(s),
          n_(L::Q * s.g.num_nodes()),
          da_(n_),
          db_(n_),
          d_flags_(s.g.num_nodes()),
          host_(static_cast<std::size_t>(n_), 0.0) {
        da_.copy_from_host(host_.data(), n_);
        db_.copy_from_host(host_.data(), n_);
        d_flags_.copy_from_host(s.flags.data(), s.g.num_nodes());
    }

    void advance(int n) {
        for (int i = 0; i < n; ++i) {
            lbm::step_gpu<L, double>(s_.g, d_flags_.data(), da_.data(), db_.data(), s_.coll);
            std::swap(da_, db_);
        }
        LBM_CUDA_CHECK(cudaDeviceSynchronize());
        da_.copy_to_host(host_.data(), n_);
    }
    const std::vector<double>& state() const { return host_; }

private:
    const ChannelSetup& s_;
    int n_;
    lbm::DeviceBuffer<double> da_, db_;
    lbm::DeviceBuffer<lbm::NodeType> d_flags_;
    std::vector<double> host_;
};

class PoiseuilleGpu : public ::testing::TestWithParam<Case> {};

TEST_P(PoiseuilleGpu, MatchesShiftedParabola) {
    const ChannelSetup s(GetParam());
    GpuStepper stepper(s);
    check_case(s, stepper);
}

INSTANTIATE_TEST_SUITE_P(Channel, PoiseuilleGpu, ::testing::ValuesIn(all_cases()),
                         [](const ::testing::TestParamInfo<Case>& info) {
                             return case_name(info.param);
                         });

}  // namespace
