#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace peryashkin_v_gauss_vstrip {

struct GaussBandInput {
  std::vector<double> augmented_matrix;  // n x (n+1), row-major
  int n{0};
  int bandwidth{0};
};

using InType = GaussBandInput;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace peryashkin_v_gauss_vstrip
