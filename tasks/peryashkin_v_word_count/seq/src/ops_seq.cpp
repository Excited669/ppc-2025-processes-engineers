#include "peryashkin_v_word_count/seq/include/ops_seq.hpp"

#include <cctype>

namespace peryashkin_v_word_count {

PeryashkinVWordCountSEQ::PeryashkinVWordCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PeryashkinVWordCountSEQ::ValidationImpl() {
  return GetOutput() == 0;
}

bool PeryashkinVWordCountSEQ::PreProcessingImpl() {
  return true;
}

bool PeryashkinVWordCountSEQ::RunImpl() {
  const auto &s = GetInput();

  int count = 0;
  bool prev_space = true;
  for (unsigned char ch : s) {
    const bool cur_space = (std::isspace(ch) != 0);
    if (!cur_space && prev_space) {
      ++count;
    }
    prev_space = cur_space;
  }

  GetOutput() = count;
  return true;
}

bool PeryashkinVWordCountSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace peryashkin_v_word_count
