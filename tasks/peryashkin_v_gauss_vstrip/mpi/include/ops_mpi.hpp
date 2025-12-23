#pragma once

#include "peryashkin_v_gauss_vstrip/common/include/common.hpp"

namespace peryashkin_v_gauss_vstrip {

class PeryashkinVGaussVStripMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PeryashkinVGaussVStripMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace peryashkin_v_gauss_vstrip
