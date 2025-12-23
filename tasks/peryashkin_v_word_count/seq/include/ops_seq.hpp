#pragma once

#include "peryashkin_v_word_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peryashkin_v_word_count {

class PeryashkinVWordCountSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PeryashkinVWordCountSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace peryashkin_v_word_count
