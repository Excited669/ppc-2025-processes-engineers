#include "peryashkin_v_gauss_vstrip/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "peryashkin_v_gauss_vstrip/common/include/common.hpp"

namespace peryashkin_v_gauss_vstrip {

namespace {

constexpr double kEpsScale = 256.0;

inline double &A(std::vector<double> &a, int n, int row, int col) {
  return a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(col)];
}
inline const double &A(const std::vector<double> &a, int n, int row, int col) {
  return a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(col)];
}

bool PivotBand(std::vector<double> &a, int n, int bw, int k, double eps) {
  int best_row = k;
  double best = std::abs(A(a, n, k, k));
  const int row_end = std::min(n - 1, k + bw);

  for (int row = k + 1; row <= row_end; ++row) {
    const double v = std::abs(A(a, n, row, k));
    if (v > best) {
      best = v;
      best_row = row;
    }
  }
  if (best <= eps) {
    return false;
  }

  if (best_row != k) {
    const int col_end = std::min(n - 1, k + bw);
    for (int col = k; col <= col_end; ++col) {
      std::swap(A(a, n, k, col), A(a, n, best_row, col));
    }
    std::swap(A(a, n, k, n), A(a, n, best_row, n));
  }
  return true;
}

void ElimBand(std::vector<double> &a, int n, int bw, int k, double eps) {
  const double diag = A(a, n, k, k);
  const int row_end = std::min(n - 1, k + bw);
  const int col_end = std::min(n - 1, k + bw);

  for (int row = k + 1; row <= row_end; ++row) {
    const double f = A(a, n, row, k) / diag;
    if (std::abs(f) <= eps) {
      continue;
    }

    A(a, n, row, k) = 0.0;
    for (int col = k + 1; col <= col_end; ++col) {
      A(a, n, row, col) -= f * A(a, n, k, col);
    }
    A(a, n, row, n) -= f * A(a, n, k, n);
  }
}

OutType BackBand(const std::vector<double> &a, int n, int bw, double eps) {
  OutType x(static_cast<std::size_t>(n), 0.0);

  for (int k = n - 1; k >= 0; --k) {
    double rhs = A(a, n, k, n);
    const int col_end = std::min(n - 1, k + bw);
    for (int col = k + 1; col <= col_end; ++col) {
      rhs -= A(a, n, k, col) * x[static_cast<std::size_t>(col)];
    }
    const double diag = A(a, n, k, k);
    if (std::abs(diag) <= eps) {
      return {};
    }
    x[static_cast<std::size_t>(k)] = rhs / diag;
  }

  return x;
}

}  // namespace

PeryashkinVGaussVStripSEQ::PeryashkinVGaussVStripSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool PeryashkinVGaussVStripSEQ::ValidationImpl() {
  const auto &in = GetInput();
  if (in.n <= 0) {
    return false;
  }
  if (in.bandwidth < 0 || in.bandwidth >= in.n) {
    return false;
  }

  const std::size_t need = static_cast<std::size_t>(in.n) * static_cast<std::size_t>(in.n + 1);
  if (in.augmented_matrix.size() != need) {
    return false;
  }

  return GetOutput().empty();
}

bool PeryashkinVGaussVStripSEQ::PreProcessingImpl() {
  return true;
}

bool PeryashkinVGaussVStripSEQ::RunImpl() {
  const auto &in = GetInput();
  const int n = in.n;
  const int bw = in.bandwidth;

  std::vector<double> a = in.augmented_matrix;
  const double eps = std::numeric_limits<double>::epsilon() * kEpsScale;

  for (int k = 0; k < n; ++k) {
    if (!PivotBand(a, n, bw, k, eps)) {
      return false;
    }
    ElimBand(a, n, bw, k, eps);
  }

  OutType x = BackBand(a, n, bw, eps);
  if (x.empty()) {
    return false;
  }

  GetOutput() = std::move(x);
  return true;
}

bool PeryashkinVGaussVStripSEQ::PostProcessingImpl() {
  return GetOutput().size() == static_cast<std::size_t>(GetInput().n);
}

}  // namespace peryashkin_v_gauss_vstrip
