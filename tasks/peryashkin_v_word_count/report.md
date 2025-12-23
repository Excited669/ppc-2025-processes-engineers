# Подсчет количества слов в строке

- **Студент**: Перяшкин Василий Андреевич 
- **Группа**: 3823Б1ПР4
- **Технология**: SEQ | MPI
- **Вариант**: 24

## 1. Введение
Подсчёт слов — простая, но показательная задача: она часто встречается в обработке текста и хорошо демонстрирует базовые идеи распараллеливания. В работе реализованы две версии решения: последовательная (SEQ) и параллельная на MPI (MPI). MPI-версия делит строку между процессами, каждый процесс считает локально, а затем результат объединяется с корректировкой пограничных случаев.

## 2. Постановка задачи
Дана строка std::string. Требуется определить число слов в строке. Под словом понимается непрерывная последовательность символов, разделённая пробелами.

Типы данных задачи:
```cpp
using InType = std::string;
using OutType = int;
```
Ограничение: входная строка должна быть непустой.

## 3. Последовательный алгоритм (SEQ)

Последовательная реализация использует один проход по строке и флаг “находимся ли мы сейчас внутри слова”.

Логика:

* когда встречаем пробел после символов слова — увеличиваем счётчик;

* когда встречаем не-пробел — отмечаем, что мы внутри слова;

* в конце, если строка завершилась внутри слова — добавляем последнее слово.

**Фрагмент кода из ops_seq.cpp:**
```cpp
int counter = 0;
bool on_word = false;
for (std::size_t i = 0; i < input.size(); i++) {
  if (input[i] == ' ' && on_word) {
    counter++;
    on_word = false;
  } else if (input[i] != ' ') {
    on_word = true;
  }
}
if (on_word) {
  counter++;
}
GetOutput() = counter;
```

**Характеристики:**

| Параметр                  | Значение |
|---------------------------|----------|
| Сложность по времени      | O(n)     |
| Сложность по памяти       | O(1)     |

## 4. Схема распараллеливания

### 4.1 Идея

В MPI-версии строка делится на size участков. Каждый процесс получает свой диапазон [begin, end) и считает слова только на своём фрагменте. Затем локальные результаты суммируются. Дополнительно обрабатываются случаи, когда слово “разорвано” между соседними процессами.

### 4.2 Разбиение строки

Фрагмент кода из ops_mpi.cpp:

```cpp
std::size_t chank = input.size() / size;
std::size_t begin = chank * rank;
std::size_t end = begin + chank;
if (rank == size - 1) {
  end = input.size();
}
```

Последний процесс получает остаток строки, если длина не делится ровно.

### 4.3 Локальный подсчёт

Каждый процесс выполняет тот же алгоритм, что и в SEQ, но только на своём диапазоне:

```cpp
int counter = 0;
bool on_word = false;
for (std::size_t i = begin; i < end; i++) {
  if (input[i] == ' ' && on_word) {
    counter++;
    on_word = false;
  } else if (input[i] != ' ') {
    on_word = true;
  }
}
```

### 4.4 Корректировка на границах

Если фрагмент начинается/заканчивается внутри слова, это фиксируется флагами. Также если конец фрагмента — не пробел, процесс добавляет слово локально (counter++), чтобы учесть незавершённое слово.

Код из ops_mpi.cpp:
```cpp
std::vector<char> start_end_with_word(2, 0);
if (rank == 0) {
  start_end_with_word.resize(2 * size, 0);
}

if (input[begin] != ' ') {
  start_end_with_word[0] = 1;
}
if (input[end - 1] != ' ') {
  counter++;
  start_end_with_word[1] = 1;
}
```

Далее:

* MPI_Reduce суммирует локальные counter в counter_sum;

* MPI_Gather собирает флаги на процессе 0;

* процесс 0 уменьшает итог на 1 для каждого “стыка”, где предыдущий кусок заканчивается словом, а следующий начинается словом (это одно слово, но посчитано дважды);

* результат рассылается всем процессам через MPI_Bcast.

Фрагмент из ops_mpi.cpp:
```cpp
MPI_Gather(start_end_with_word.data(), 2, MPI_CHAR,
           start_end_with_word.data(), 2, MPI_CHAR, 0, MPI_COMM_WORLD);

int counter_sum = 0;
MPI_Reduce(&counter, &counter_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

if (rank == 0) {
  for (int i = 1; i < size; i++) {
    if ((start_end_with_word[i * 2 - 1] == 1) && (start_end_with_word[i * 2] == 1)) {
      counter_sum--;
    }
  }
}

MPI_Bcast(&counter_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
GetOutput() = counter_sum;

```
### Схема работы программы 
```
┌──────────────────────────────┐
│        Входная строка        │
└────────┬─────────────────────┘
         │ (делим на части)
         ↓
 ┌──────────────┬──────────────┬──────────────┐
 │  Процесс 0   │  Процесс 1   │  Процесс 2   │  ...
 │   часть 0    │   часть 1    │   часть 2    │
 └───────┬──────┴───────┬──────┴───────┬──────┘
         │              │              │
         ↓              ↓              ↓
   local_count + flags  local_count + flags  local_count + flags
         └──────────────┬──────────────┘
                        ↓
           MPI_Reduce (сумма счётчиков)
           MPI_Gather (сбор флагов на rank=0)
                        ↓
      rank=0 корректирует двойной счёт на границах
                        ↓
                  MPI_Bcast (ответ всем)

```

## 5. Детали реализации

**Структура проекта**
| Файл                         | Назначение                               |
| ---------------------------- | ---------------------------------------- |
| `common.hpp`                 | типы входа/выхода и базовый класс задачи |
| `ops_seq.hpp/.cpp`           | последовательное решение                 |
| `ops_mpi.hpp/.cpp`           | MPI-решение                              |
| `tests/functional/main.cpp`  | функциональные тесты                     |
| `tests/performance/main.cpp` | тесты производительности                 |


### common.hpp
```cpp
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
```

### ops_seq.hpp
```cpp
#pragma once

#include "peryashkin_v_word_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peryashkin_v_word_count {

class PeryashkinVWordCountSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit PeryashkinVWordCountSEQ(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace peryashkin_v_word_count
```

### ops_seq.cpp
```cpp
#include "peryashkin_v_word_count/seq/include/ops_seq.hpp"

#include <cstddef>
#include <string>

#include "peryashkin_v_word_count/common/include/common.hpp"

namespace peryashkin_v_word_count {

PeryashkinVWordCountSEQ::PeryashkinVWordCountSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PeryashkinVWordCountSEQ::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool PeryashkinVWordCountSEQ::PreProcessingImpl() { return true; }

bool PeryashkinVWordCountSEQ::RunImpl() {
  const std::string input = GetInput();
  if (input.empty()) return false;

  int counter = 0;
  bool on_word = false;

  for (std::size_t i = 0; i < input.size(); ++i) {
    if (input[i] == ' ' && on_word) {
      ++counter;
      on_word = false;
    } else if (input[i] != ' ') {
      on_word = true;
    }
  }

  if (on_word) ++counter;

  GetOutput() = counter;
  return true;
}

bool PeryashkinVWordCountSEQ::PostProcessingImpl() { return true; }

}  // namespace peryashkin_v_word_count
```

### ops_mpi.hpp
```cpp
#pragma once

#include "peryashkin_v_word_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace peryashkin_v_word_count {

class PeryashkinVWordCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit PeryashkinVWordCountMPI(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace peryashkin_v_word_count
```

### ops_mpi.cpp
```cpp
#include "peryashkin_v_word_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "peryashkin_v_word_count/common/include/common.hpp"

namespace peryashkin_v_word_count {

namespace {

std::pair<int, std::array<char, 2>> CountWordsLocal(const std::string& s) {
  int counter = 0;
  bool on_word = false;

  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == ' ' && on_word) {
      ++counter;
      on_word = false;
    } else if (s[i] != ' ') {
      on_word = true;
    }
  }

  std::array<char, 2> flags = {0, 0};

  if (!s.empty() && s.front() != ' ') flags[0] = 1;
  if (!s.empty() && s.back()  != ' ') {
    ++counter;        
    flags[1] = 1;
  }

  return {counter, flags};
}

void BuildScatterMeta(const std::string& input, int size,
                      std::vector<int>& counts, std::vector<int>& displs) {
  const std::size_t chunk = input.size() / static_cast<std::size_t>(size);
  for (int i = 0; i < size; ++i) {
    displs[i] = static_cast<int>(chunk * static_cast<std::size_t>(i));
    counts[i] = (i == size - 1) ? static_cast<int>(input.size()) - displs[i]
                                : static_cast<int>(chunk);
  }
}

void FixDoubleCountOnBorders(const std::vector<char>& all_flags, int size, int& sum) {
  for (int i = 1; i < size; ++i) {
    const std::size_t prev_end = static_cast<std::size_t>(i) * 2U - 1U;
    const std::size_t curr_beg = prev_end + 1U;
    if (all_flags[prev_end] == 1 && all_flags[curr_beg] == 1) {
      --sum;
    }
  }
}

}  // namespace

PeryashkinVWordCountMPI::PeryashkinVWordCountMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PeryashkinVWordCountMPI::ValidationImpl() {
  return (!GetInput().empty()) && (GetOutput() == 0);
}

bool PeryashkinVWordCountMPI::PreProcessingImpl() { return true; }

bool PeryashkinVWordCountMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string input;
  std::vector<int> send_counts(size, 0);
  std::vector<int> send_displs(size, 0);

  if (rank == 0) {
    input = GetInput();
    if (input.empty()) return false;
    BuildScatterMeta(input, size, send_counts, send_displs);
  }

  MPI_Bcast(send_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(send_displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  const int local_size = send_counts[rank];
  std::vector<char> local_buf(local_size);

  std::vector<char> input_buf;
  if (rank == 0) input_buf.assign(input.begin(), input.end());

  MPI_Scatterv(rank == 0 ? input_buf.data() : nullptr,
               send_counts.data(), send_displs.data(), MPI_CHAR,
               local_buf.data(), local_size, MPI_CHAR,
               0, MPI_COMM_WORLD);

  const std::string local_str(local_buf.begin(), local_buf.end());

  auto [local_count, local_flags] = CountWordsLocal(local_str);

  int global_sum = 0;
  MPI_Reduce(&local_count, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  std::vector<char> all_flags;
  if (rank == 0) all_flags.assign(static_cast<std::size_t>(2) * size, 0);

  MPI_Gather(local_flags.data(), 2, MPI_CHAR,
             rank == 0 ? all_flags.data() : nullptr, 2, MPI_CHAR,
             0, MPI_COMM_WORLD);

  if (rank == 0) {
    FixDoubleCountOnBorders(all_flags, size, global_sum);
  }

  MPI_Bcast(&global_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_sum;
  return true;
}

bool PeryashkinVWordCountMPI::PostProcessingImpl() { return true; }

}  // namespace peryashkin_v_word_count
```

## 6. Экспериментальная среда

| Компонент  | Значение                                     |
| ---------- | -------------------------------------------- |
| CPU        | Apple M2 (8 cores)                           |
| RAM        | 8 GB (хост), в контейнере доступно ~4.8 GiB |
| ОС         | macOS + Ubuntu 24.04 (DevContainer)          |
| Компилятор | GCC / g++ (CMake, Release)                   |
| MPI        | Open MPI 4.1.6                               |


Тестовые данные формируются псевдослучайно:

* каждое слово собирается из строчных латинских букв в диапазоне 'a'..'z';
* длина одного слова выбирается случайно и лежит в пределах от 1 до 10 символов;
* слова в строке разделяются пробелами (перед каждым словом добавляется пробел).

## 7. Результаты и обсуждение

### 7.1 Корректность
Корректность проверялась набором функциональных тестов на строках с 1, 7 и 1000 словами. Результаты показали, что реализации SEQ и MPI возвращают одинаково верный ответ во всех случаях.

### 7.2 Производительность
Используются стандартные метрики:

* Ускорение: Speedup = T_seq / T_parallel

* Эффективность: Efficiency = Speedup / Count * 100%

Ниже приведены измерения для моей задачи (время взято из вывода perf-тестов).
Показаны два режима:

* task_run — “чистое” время выполнения задачи

* pipeline — полное время с накладными расходами (инициализация/взаимодействие)

Измерения времени task_run
| Mode | Count | Time, ms | Speedup | Efficiency |
| ---- | ----- | -------- | ------- | ---------- |
| seq  | 1     | 7.722    | 1.00    | N/A        |
| mpi  | 4     | 6.831    | 1.13    | 28.3%      |
| mpi  | 8     | 13.771   | 0.56    | 7.0%       |
| mpi  | 12    | 15.163   | 0.51    | 4.2%       |
| mpi  | 20    | 74.127   | 0.10    | 0.5%       |

Полное время выполнения pipeline
| Mode | Count | Time, ms | Speedup | Efficiency |
| ---- | ----- | -------- | ------- | ---------- |
| seq  | 1     | 7.777    | 1.00    | N/A        |
| mpi  | 4     | 5.363    | 1.45    | 36.3%      |
| mpi  | 8     | 23.344   | 0.33    | 4.2%       |
| mpi  | 12    | 33.041   | 0.24    | 2.0%       |
| mpi  | 20    | 40.676   | 0.19    | 1.0%       |

На небольшом числе процессов (например, 4) MPI может дать ускорение, но при увеличении Count накладные расходы на коммуникацию и синхронизацию начинают доминировать. Дополнительно при --oversubscribe процессы конкурируют за ресурсы CPU, из-за чего время может заметно ухудшаться.

## 8. Заключение

В ходе работы:

* реализован последовательный алгоритм подсчёта слов в строке;
разработана MPI-версия с разбиением строки на части и корректировкой “разорванных” слов на границах;
* написаны функциональные и performance тесты;
* проведено сравнение производительности.

MPI-реализация корректна и демонстрирует ускорение на умеренном числе процессов, однако при сильном увеличении количества процессов эффективность падает из-за накладных расходов.

## 9. Источники
1. Документация Open MPI
2. MPI Standard (описание коллективных операций: MPI_Bcast, MPI_Scatterv, MPI_Gather, MPI_Reduce)

3. Материалы курса по параллельному программированию (сборка, структура задач, тестирование)
