#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "peryashkin_v_gauss_vstrip/common/include/common.hpp"
#include "peryashkin_v_gauss_vstrip/mpi/include/ops_mpi.hpp"
#include "peryashkin_v_gauss_vstrip/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace peryashkin_v_gauss_vstrip {

namespace {

GaussBandInput MakePerfSystem(int n, int bandwidth) {
  GaussBandInput input;
  input.n = n;
  input.bandwidth = bandwidth;
  input.augmented_matrix.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1), 0.0);

  std::vector<double> x_true(static_cast<std::size_t>(n), 1.0);

  auto idx = [&](int r, int c) -> std::size_t {
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(n + 1) + static_cast<std::size_t>(c);
  };

  for (int i = 0; i < n; ++i) {
    for (int j = std::max(0, i - bandwidth); j <= std::min(n - 1, i + bandwidth); ++j) {
      input.augmented_matrix[idx(i, j)] = (i == j) ? static_cast<double>(2 * bandwidth + 1) : 1.0;
    }
  }

  for (int i = 0; i < n; ++i) {
    double b_val = 0.0;
    for (int j = std::max(0, i - bandwidth); j <= std::min(n - 1, i + bandwidth); ++j) {
      b_val += input.augmented_matrix[idx(i, j)] * x_true[static_cast<std::size_t>(j)];
    }
    input.augmented_matrix[idx(i, n)] = b_val;
  }

  return input;
}

}  // namespace

class PeryashkinVGaussVStripPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int n = 2000;
    const int bw = 5;
    input_data_ = MakePerfSystem(n, bw);
    expected_output_.assign(static_cast<std::size_t>(n), 1.0);
  }

  bool CheckTestOutputData(OutType& output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    const double eps = 1e-8;
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (std::fabs(output_data[i] - expected_output_[i]) > eps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(PeryashkinVGaussVStripPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PeryashkinVGaussVStripMPI, PeryashkinVGaussVStripSEQ>(
        PPC_SETTINGS_peryashkin_v_gauss_vstrip);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(PeryashkinVGaussVStripPerf, PeryashkinVGaussVStripPerfTests, kGtestValues,
                         PeryashkinVGaussVStripPerfTests::CustomPerfTestName);

}  // namespace peryashkin_v_gauss_vstrip
