#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <string>

#include "peryashkin_v_word_count/common/include/common.hpp"
#include "peryashkin_v_word_count/mpi/include/ops_mpi.hpp"
#include "peryashkin_v_word_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace peryashkin_v_word_count {

class PeryashkinVWordCountRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int seed = 123;
    correct_ = 1000000;

    // NOLINTNEXTLINE(cert-msc51-cpp)
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    std::uniform_int_distribution<int> word_len(1, 10);
    std::uniform_int_distribution<int> chars('a', 'z');

    std::string s;
    s.reserve(static_cast<std::size_t>(correct_) * 12U);

    for (int i = 0; i < correct_; ++i) {
      s.push_back(' ');
      const int len = word_len(gen);
      for (int j = 0; j < len; ++j) {
        s.push_back(static_cast<char>(chars(gen)));
      }
    }

    input_ = s;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == correct_;
  }
  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
  int correct_ = 0;
};

TEST_P(PeryashkinVWordCountRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PeryashkinVWordCountMPI, PeryashkinVWordCountSEQ>(
    PPC_SETTINGS_peryashkin_v_word_count);

const auto kValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kName = PeryashkinVWordCountRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PeryashkinVWordCountPerf, PeryashkinVWordCountRunPerfTests, kValues, kName);

}  // namespace peryashkin_v_word_count
