// Poiseuille validation (CPU, double precision): the body-force-driven channel
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

class CpuStepper {
public:
    explicit CpuStepper(const ChannelSetup& s)
        : s_(s), a_(static_cast<std::size_t>(L::Q * s.g.num_nodes()), 0.0), b_(a_) {}

    void advance(int n) {
        for (int i = 0; i < n; ++i) {
            lbm::step_cpu<L, double>(s_.g, s_.flags.data(), a_.data(), b_.data(), s_.coll);
            std::swap(a_, b_);
        }
    }
    const std::vector<double>& state() const { return a_; }

private:
    const ChannelSetup& s_;
    std::vector<double> a_, b_;
};

class PoiseuilleCpu : public ::testing::TestWithParam<Case> {};

TEST_P(PoiseuilleCpu, MatchesShiftedParabola) {
    const ChannelSetup s(GetParam());
    CpuStepper stepper(s);
    check_case(s, stepper);
}

INSTANTIATE_TEST_SUITE_P(Channel, PoiseuilleCpu, ::testing::ValuesIn(all_cases()),
                         [](const ::testing::TestParamInfo<Case>& info) {
                             return case_name(info.param);
                         });

}  // namespace
