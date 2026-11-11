// Poiseuille validation (CPU, double and single precision): the body-force-driven channel
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

template <typename T>
class CpuStepper {
public:
    explicit CpuStepper(const ChannelSetup<T>& s)
        : s_(s), a_(static_cast<std::size_t>(L::Q * s.g.num_nodes()), T(0)), b_(a_) {}

    void advance(int n) {
        for (int i = 0; i < n; ++i) {
            lbm::step_cpu<L, T>(s_.g, s_.flags.data(), a_.data(), b_.data(), s_.coll);
            std::swap(a_, b_);
        }
    }
    const std::vector<T>& state() const { return a_; }

private:
    const ChannelSetup<T>& s_;
    std::vector<T> a_, b_;
};

template <typename T>
void run_case(const Case& c) {
    const ChannelSetup<T> s(c);
    CpuStepper<T> stepper(s);
    check_case(s, stepper);
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

}  // namespace
