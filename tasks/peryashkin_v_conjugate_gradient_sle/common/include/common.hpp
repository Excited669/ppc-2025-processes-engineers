#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace peryashkin_v_conjugate_gradient_sle {

using InType = std::pair<int, int>;               // (n, variant)
using OutType = std::vector<double>;              // solution vector x
using TestType = std::tuple<InType, std::string>; // (input, label)
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace peryashkin_v_conjugate_gradient_sle
