#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>

#include "peryashkin_v_word_count/common/include/common.hpp"
#include "peryashkin_v_word_count/mpi/include/ops_mpi.hpp"
#include "peryashkin_v_word_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace peryashkin_v_word_count {

class PeryashkinVWordCountRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType& p) { return std::get<0>(p); }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::mt19937 gen(std::get<1>(params));
    std::uniform_int_distribution<> word_len(1, 10);
    std::uniform_int_distribution<> chars('a', 'z');

    correct_ = std::get<2>(params);

    std::string s;
    s.reserve(static_cast<std::size_t>(correct_) * 12U);

    for (int i = 0; i < correct_; ++i) {
      s.push_back(' ');
      const int len = word_len(gen);
      for (int j = 0; j < len; ++j) s.push_back(static_cast<char>(chars(gen)));
    }
    input_ = s;
  }

  bool CheckTestOutputData(OutType& output_data) final { return output_data == correct_; }
  InType GetTestInputData() final { return input_; }

 private:
  InType input_;
  int correct_ = 0;
};

namespace {

TEST_P(PeryashkinVWordCountRunFuncTests, WordsCounting) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple("Gen_1_word_seed_123", 123, 1),
    std::make_tuple("Gen_7_word_seed_123", 123, 7),
    std::make_tuple("Gen_1000_word_seed_123", 123, 1000)};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<PeryashkinVWordCountMPI, InType>(kTestParam, PPC_SETTINGS_peryashkin_v_word_count),
    ppc::util::AddFuncTask<PeryashkinVWordCountSEQ, InType>(kTestParam, PPC_SETTINGS_peryashkin_v_word_count));

const auto kValues = ppc::util::ExpandToValues(kTasks);
const auto kName = PeryashkinVWordCountRunFuncTests::PrintFuncTestName<PeryashkinVWordCountRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(PeryashkinVWordCount, PeryashkinVWordCountRunFuncTests, kValues, kName);

}  // namespace

}  // namespace peryashkin_v_word_count
