#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "peryashkin_v_conjugate_gradient_sle/common/include/common.hpp"
#include "peryashkin_v_conjugate_gradient_sle/mpi/include/ops_mpi.hpp"
#include "peryashkin_v_conjugate_gradient_sle/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace peryashkin_v_conjugate_gradient_sle {

class PeryashkinVConjGradSlePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kN = 3500;
  InType input_data_{kN, 0};

  void SetUp() override {
    input_data_ = {kN, 0};
  }

  bool CheckTestOutputData(OutType& output_data) override {
    int rank = 0;
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (mpi_init) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank > 0) return true;
    if (static_cast<int>(output_data.size()) != kN) return false;

    double max_diff = 0.0;
    for (int i = 0; i < kN; ++i) {
      double ax_i = 4.0 * output_data[static_cast<std::size_t>(i)];
      if (i > 0) ax_i += output_data[static_cast<std::size_t>(i - 1)];
      if (i + 1 < kN) ax_i += output_data[static_cast<std::size_t>(i + 1)];
      max_diff = std::max(max_diff, std::fabs(ax_i - 1.0));
    }
    return max_diff < 1e-6;
  }

  InType GetTestInputData() override { return input_data_; }
};

TEST_P(PeryashkinVConjGradSlePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PeryashkinVConjGradSleMPI, PeryashkinVConjGradSleSEQ>(
        PPC_SETTINGS_peryashkin_v_conjugate_gradient_sle);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = PeryashkinVConjGradSlePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunMode, PeryashkinVConjGradSlePerfTests, kGtestValues, kPerfTestName);

}  // namespace peryashkin_v_conjugate_gradient_sle
