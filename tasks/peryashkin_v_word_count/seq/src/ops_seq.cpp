#include "peryashkin_v_word_count/seq/include/ops_seq.hpp"

#include <cctype>
#include <string>

#include "peryashkin_v_word_count/common/include/common.hpp"

namespace peryashkin_v_word_count {

PeryashkinVWordCountSEQ::PeryashkinVWordCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PeryashkinVWordCountSEQ::ValidationImpl() {
  // Пустая строка допустима
  return GetOutput() == 0;
}

bool PeryashkinVWordCountSEQ::PreProcessingImpl() {
  return true;
}

bool PeryashkinVWordCountSEQ::RunImpl() {
  const std::string &s = GetInput();
  if (s.empty()) {
    GetOutput() = 0;
    return true;
  }

  int words = 0;
  bool in_word = false;

  for (char ch : s) {
    const bool space = (std::isspace(static_cast<unsigned char>(ch)) != 0);
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
