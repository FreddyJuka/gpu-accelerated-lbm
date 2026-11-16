// Poiseuille validation (CPU; double, single precision, and 16-bit storage
// with single-precision compute): the body-force-driven channel
// with halfway bounce-back reaches the analytical parabola shifted by the
// exact TRT slip. ChannelSetup, reference, measurement and tolerance are documented
// in poiseuille_common.hpp.

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "lbm/cpu/step.hpp"
#include "poiseuille_common.hpp"

namespace {

using namespace poiseuille;

// T is the compute type, Storage the type of the population buffers (T by
// default). The rest state is the value-initialised Storage: 0.0, or the
// all-zero P16 pattern (+0).
template <typename T, typename Storage = T>
class CpuStepper {
public:
    explicit CpuStepper(const ChannelSetup<T>& s)
        : s_(s), a_(static_cast<std::size_t>(L::Q * s.g.num_nodes()), Storage{}), b_(a_) {}

    void advance(int n) {
        for (int i = 0; i < n; ++i) {
            lbm::step_cpu<L, Storage>(s_.g, s_.flags.data(), a_.data(), b_.data(), s_.coll);
            std::swap(a_, b_);
        }
    }
    const std::vector<Storage>& state() const { return a_; }

private:
    const ChannelSetup<T>& s_;
    std::vector<Storage> a_, b_;
};

template <typename T>
void run_case(const Case& c) {
    const ChannelSetup<T> s(c);
    CpuStepper<T> stepper(s);
    check_case(s, stepper);
}

void run_case_p16(const Case& c, SlipClaim claim) {
    const ChannelSetup<float> s(c);
    CpuStepper<float, lbm::P16> stepper(s);
    check_case(s, stepper, claim);
}

auto param_name = [](const ::testing::TestParamInfo<Case>& info) { return case_name(info.param); };

class PoiseuilleCpu : public ::testing::TestWithParam<Case> {};

TEST_P(PoiseuilleCpu, MatchesShiftedParabola) { run_case<double>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Channel, PoiseuilleCpu, ::testing::ValuesIn(all_cases()), param_name);

// Single precision (the production precision): the cases of float_cases(),
// see "Single precision" in poiseuille_common.hpp.
class PoiseuilleCpuFloat : public ::testing::TestWithParam<Case> {};

TEST_P(PoiseuilleCpuFloat, MatchesShiftedParabola) { run_case<float>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Channel, PoiseuilleCpuFloat, ::testing::ValuesIn(float_cases()),
                         param_name);

// 16-bit storage (lbm::P16) with float compute: the cases of p16_cases(). Only
// H = 8 with nu = 0.5 resolves the slip; the other cases check the profile
// against u_a + du within the derived tolerance without claiming to resolve
// it. See "16-bit storage" in poiseuille_common.hpp.
class PoiseuilleCpuP16 : public ::testing::TestWithParam<Case> {};

TEST_P(PoiseuilleCpuP16, MatchesShiftedParabola) { run_case_p16(GetParam(), p16_slip_claim(GetParam())); }

INSTANTIATE_TEST_SUITE_P(Channel, PoiseuilleCpuP16, ::testing::ValuesIn(p16_cases()), param_name);

}  // namespace
