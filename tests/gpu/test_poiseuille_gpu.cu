// Poiseuille validation on the GPU: same cases, checks and bounds as
// test_poiseuille.cpp (see poiseuille_common.hpp). This target keeps nvcc's
// default FMA contraction (production flags). That the tolerance derivation
// also holds for fused arithmetic is asserted, not argued.

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "../poiseuille_common.hpp"
#include "lbm/cuda/check.hpp"
#include "lbm/cuda/device_buffer.hpp"
#include "lbm/cuda/step.cuh"

namespace {

using namespace poiseuille;

// T is the compute type, Storage the type of the population buffers (T by
// default). The rest state is the value-initialised Storage: 0.0, or the
// all-zero P16 pattern (+0).
template <typename T, typename Storage = T>
class GpuStepper {
public:
    explicit GpuStepper(const ChannelSetup<T>& s)
        : s_(s),
          n_(L::Q * s.g.num_nodes()),
          da_(n_),
          db_(n_),
          d_flags_(s.g.num_nodes()),
          host_(static_cast<std::size_t>(n_), Storage{}) {
        da_.copy_from_host(host_.data(), n_);
        db_.copy_from_host(host_.data(), n_);
        d_flags_.copy_from_host(s.flags.data(), s.g.num_nodes());
    }

    void advance(int n) {
        for (int i = 0; i < n; ++i) {
            lbm::step_gpu<L, Storage>(s_.g, d_flags_.data(), da_.data(), db_.data(), s_.coll);
            std::swap(da_, db_);
        }
        LBM_CUDA_CHECK(cudaDeviceSynchronize());
        da_.copy_to_host(host_.data(), n_);
    }
    const std::vector<Storage>& state() const { return host_; }

private:
    const ChannelSetup<T>& s_;
    int n_;
    lbm::DeviceBuffer<Storage> da_, db_;
    lbm::DeviceBuffer<lbm::NodeType> d_flags_;
    std::vector<Storage> host_;
};

template <typename T>
void run_case(const Case& c) {
    const ChannelSetup<T> s(c);
    GpuStepper<T> stepper(s);
    check_case(s, stepper);
}

void run_case_p16(const Case& c, SlipClaim claim) {
    const ChannelSetup<float> s(c);
    GpuStepper<float, lbm::P16> stepper(s);
    check_case(s, stepper, claim);
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

// 16-bit storage (lbm::P16) with float compute: the cases of p16_cases(). Only
// H = 8 with nu = 0.5 resolves the slip; the other cases check the profile
// against u_a + du within the derived tolerance without claiming to resolve
// it. See "16-bit storage" in poiseuille_common.hpp.
class PoiseuilleGpuP16 : public ::testing::TestWithParam<Case> {};

TEST_P(PoiseuilleGpuP16, MatchesShiftedParabola) { run_case_p16(GetParam(), p16_slip_claim(GetParam())); }

INSTANTIATE_TEST_SUITE_P(Channel, PoiseuilleGpuP16, ::testing::ValuesIn(p16_cases()), param_name);

}  // namespace
