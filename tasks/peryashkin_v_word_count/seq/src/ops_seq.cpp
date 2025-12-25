#include "peryashkin_v_word_count/seq/include/ops_seq.hpp"

#include <cctype>
#include <cstddef>
#include <string>

#include "peryashkin_v_word_count/common/include/common.hpp"

namespace peryashkin_v_word_count {

PeryashkinVWordCountSEQ::PeryashkinVWordCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PeryashkinVWordCountSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool PeryashkinVWordCountSEQ::PreProcessingImpl() {
  return true;
}

bool PeryashkinVWordCountSEQ::RunImpl() {
  const std::string &s = GetInput();
  if (s.empty()) {
    return false;
  }

  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };

  int words = 0;
  bool in_word = false;
  for (char ch : s) {
    const bool space = is_space(static_cast<unsigned char>(ch));
    if (space) {
      if (in_word) {
        ++words;
        in_word = false;
      }
    } else {
      in_word = true;
    }
  }
  if (in_word) {
    ++words;
  }

  GetOutput() = words;
  return true;
}

bool PeryashkinVWordCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace peryashkin_v_word_count
