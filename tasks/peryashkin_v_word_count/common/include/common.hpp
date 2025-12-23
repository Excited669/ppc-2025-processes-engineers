#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace peryashkin_v_word_count {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<std::string, int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace peryashkin_v_word_count
