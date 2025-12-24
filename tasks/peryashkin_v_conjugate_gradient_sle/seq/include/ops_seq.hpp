#pragma once

#include "peryashkin_v_conjugate_gradient_sle/common/include/common.hpp"

namespace peryashkin_v_conjugate_gradient_sle {

class PeryashkinVConjGradSleSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit PeryashkinVConjGradSleSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace peryashkin_v_conjugate_gradient_sle
