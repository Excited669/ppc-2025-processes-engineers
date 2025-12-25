#include "peryashkin_v_word_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

#include "peryashkin_v_word_count/common/include/common.hpp"

namespace peryashkin_v_word_count {

namespace {

// local_count считает "начала слов" внутри чанка, считая что перед чанком пробел.
// затем на rank 0 мы исправим границы между чанками.
struct LocalResult {
  int local_count = 0;
  char starts_with_word = 0;
  char ends_with_word = 0;
};

LocalResult CountLocal(const std::string &part) {
  LocalResult r{};
  if (part.empty()) {
    return r;
  }

  bool prev_space = true;
  for (char ch : part) {
    const bool is_space = (std::isspace(static_cast<unsigned char>(ch)) != 0);
    if (!is_space && prev_space) {
      ++r.local_count;
    }
    prev_space = is_space;
  }

  const bool starts = (std::isspace(static_cast<unsigned char>(part.front())) == 0);
  const bool ends = (std::isspace(static_cast<unsigned char>(part.back())) == 0);
  r.starts_with_word = static_cast<char>(starts ? 1 : 0);
  r.ends_with_word = static_cast<char>(ends ? 1 : 0);

  return r;
}

void MakeScatterPlan(int n, int size, std::vector<int> &counts, std::vector<int> &displs) {
  counts.assign(size, 0);
  displs.assign(size, 0);

  const int base = (size == 0) ? 0 : (n / size);
  const int rem = (size == 0) ? 0 : (n % size);

  int offset = 0;
  for (int i = 0; i < size; ++i) {
    const int add = (i < rem) ? 1 : 0;
    counts[i] = base + add;
    displs[i] = offset;
    offset += counts[i];
  }
}

}  // namespace

PeryashkinVWordCountMPI::PeryashkinVWordCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PeryashkinVWordCountMPI::ValidationImpl() {
  // Пустая строка допустима
  return GetOutput() == 0;
}

bool PeryashkinVWordCountMPI::PreProcessingImpl() {
  return true;
}

bool PeryashkinVWordCountMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string input;
  int n = 0;

  std::vector<int> send_counts;
  std::vector<int> send_displs;

  if (rank == 0) {
    input = GetInput();
    n = static_cast<int>(input.size());
    MakeScatterPlan(n, size, send_counts, send_displs);
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    send_counts.resize(size);
    send_displs.resize(size);
  }
  MPI_Bcast(send_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(send_displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  const int local_n = send_counts[rank];
  std::vector<char> local_buf(static_cast<std::size_t>(local_n));

  const char *send_ptr = nullptr;
  if (rank == 0 && n > 0) {
    send_ptr = input.data();
  }

  MPI_Scatterv(send_ptr, send_counts.data(), send_displs.data(), MPI_CHAR, local_buf.data(), local_n, MPI_CHAR, 0,
               MPI_COMM_WORLD);

  const std::string local_part(local_buf.begin(), local_buf.end());
  const LocalResult lr = CountLocal(local_part);

  int sum = 0;
  MPI_Reduce(&lr.local_count, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  // Соберём флаги границ, чтобы скорректировать слова, разорванные по чанкам
  std::vector<char> all_flags;
  if (rank == 0) {
    const std::size_t flags_len = static_cast<std::size_t>(size) * 2U;
    all_flags.resize(flags_len, 0);
  }

  const std::array<char, 2> my_flags = {lr.starts_with_word, lr.ends_with_word};
  MPI_Gather(my_flags.data(), 2, MPI_CHAR, (rank == 0) ? all_flags.data() : nullptr, 2, MPI_CHAR, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    // Если предыдущий чанк заканчивается внутри слова и следующий начинается внутри слова,
    // то одно слово посчитали дважды -> -1
    for (int i = 1; i < size; ++i) {
      const std::size_t prev_end_idx = (static_cast<std::size_t>(i - 1) * 2U) + 1U;  // <-- FIX: скобки
      const std::size_t cur_begin_idx = (static_cast<std::size_t>(i) * 2U);

      const char prev_end = all_flags[prev_end_idx];
      const char cur_begin = all_flags[cur_begin_idx];
      if (prev_end == 1 && cur_begin == 1) {
        --sum;
      }
    }
    GetOutput() = sum;
  }

  MPI_Bcast(&sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = sum;

  return true;
}

bool PeryashkinVWordCountMPI::PostProcessingImpl() {
  return true;
}

}  // namespace peryashkin_v_word_count
