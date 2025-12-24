#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "peryashkin_v_conjugate_gradient_sle/common/include/common.hpp"
#include "peryashkin_v_conjugate_gradient_sle/mpi/include/ops_mpi.hpp"
#include "peryashkin_v_conjugate_gradient_sle/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace peryashkin_v_conjugate_gradient_sle {

class PeryashkinVConjGradSleFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType& test_param) {
    const auto in = std::get<0>(test_param);
    return std::to_string(in.first) + "_" + std::to_string(in.second) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType& output_data) override {
    int rank = 0;
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (mpi_init) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // In MPI mode only rank 0 owns the collected answer
    if (rank > 0) return true;

    const int n = input_data_.first;
    const int variant = input_data_.second;
    if (static_cast<int>(output_data.size()) != n) return false;

    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
      double ax_i = 0.0;

      if (variant == 1) {
        ax_i = 5.0 * output_data[static_cast<std::size_t>(i)];
      } else if (variant == 0) {
        ax_i = 4.0 * output_data[static_cast<std::size_t>(i)];
        if (i > 0) ax_i += output_data[static_cast<std::size_t>(i - 1)];
        if (i + 1 < n) ax_i += output_data[static_cast<std::size_t>(i + 1)];
      } else {  // variant == 2
        ax_i = 3.0 * output_data[static_cast<std::size_t>(i)];
        const int j = n - 1 - i;
        if (j != i) ax_i -= output_data[static_cast<std::size_t>(j)];
      }

      max_diff = std::max(max_diff, std::fabs(ax_i - 1.0));
    }

    return max_diff < 1e-6;
  }

  InType GetTestInputData() override { return input_data_; }

 private:
  InType input_data_{0, 0};
};

namespace {

TEST_P(PeryashkinVConjGradSleFuncTests, ConjugateGradientSolvesAxEqOnes) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {
    std::make_tuple(std::make_pair(1, 0), "n1_v0"),
    std::make_tuple(std::make_pair(3, 0), "n3_v0"),
    std::make_tuple(std::make_pair(10, 0), "n10_v0"),
    std::make_tuple(std::make_pair(50, 0), "n50_v0"),
    std::make_tuple(std::make_pair(1, 1), "n1_v1"),
    std::make_tuple(std::make_pair(3, 1), "n3_v1"),
    std::make_tuple(std::make_pair(10, 1), "n10_v1"),
    std::make_tuple(std::make_pair(50, 1), "n50_v1"),
    std::make_tuple(std::make_pair(1, 2), "n1_v2"),
    std::make_tuple(std::make_pair(3, 2), "n3_v2"),
    std::make_tuple(std::make_pair(10, 2), "n10_v2"),
    std::make_tuple(std::make_pair(50, 2), "n50_v2"),
};

const auto kTestTasksList =
    std::tuple_cat(
        ppc::util::AddFuncTask<PeryashkinVConjGradSleMPI, InType>(kTestParam,
                                                                 PPC_SETTINGS_peryashkin_v_conjugate_gradient_sle),
        ppc::util::AddFuncTask<PeryashkinVConjGradSleSEQ, InType>(kTestParam,
                                                                 PPC_SETTINGS_peryashkin_v_conjugate_gradient_sle));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    PeryashkinVConjGradSleFuncTests::PrintFuncTestName<PeryashkinVConjGradSleFuncTests>;

INSTANTIATE_TEST_SUITE_P(Basic, PeryashkinVConjGradSleFuncTests, kGtestValues, kFuncTestName);

// Some quick edge checks (SEQ & MPI validation sanity)
TEST(PeryashkinVConjGradSleEdge, InvalidNSeq) {
  PeryashkinVConjGradSleSEQ t({0, 0});
  EXPECT_FALSE(t.Validation());
}

TEST(PeryashkinVConjGradSleEdge, InvalidVariantSeq) {
  PeryashkinVConjGradSleSEQ t({10, -1});
  EXPECT_FALSE(t.Validation());
}

TEST(PeryashkinVConjGradSleEdge, InvalidNMpi) {
  PeryashkinVConjGradSleMPI t({0, 0});
  EXPECT_FALSE(t.Validation());
}

TEST(PeryashkinVConjGradSleEdge, InvalidVariantMpi) {
  PeryashkinVConjGradSleMPI t({10, 99});
  EXPECT_FALSE(t.Validation());
}

}  // namespace

}  // namespace peryashkin_v_conjugate_gradient_sle
