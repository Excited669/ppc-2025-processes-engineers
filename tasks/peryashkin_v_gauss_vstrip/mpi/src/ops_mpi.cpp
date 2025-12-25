#include "peryashkin_v_gauss_vstrip/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "peryashkin_v_gauss_vstrip/common/include/common.hpp"

namespace peryashkin_v_gauss_vstrip {

namespace {

constexpr double kEpsScale = 256.0;

struct Layout {
  int n{};
  int m{};  // n + 1 (augmented cols)
  int world_size{};
  int world_rank{};
  int my_first_col{};
  int my_cols{};

  std::vector<int> first_col;
  std::vector<int> cols;
  std::vector<int> sendcounts;
  std::vector<int> displs;
};

[[nodiscard]] Layout MakeLayout(int n, int world_size, int world_rank) {
  Layout layout;
  layout.n = n;
  layout.m = n + 1;
  layout.world_size = world_size;
  layout.world_rank = world_rank;

  layout.first_col.assign(world_size, 0);
  layout.cols.assign(world_size, 0);
  layout.sendcounts.assign(world_size, 0);
  layout.displs.assign(world_size, 0);

  const int base = layout.m / world_size;
  const int rem = layout.m % world_size;

  int cur = 0;
  for (int proc = 0; proc < world_size; ++proc) {
    const int block_cols = base + ((proc < rem) ? 1 : 0);
    layout.first_col[proc] = cur;
    layout.cols[proc] = block_cols;
    cur += block_cols;
  }

  int disp = 0;
  for (int proc = 0; proc < world_size; ++proc) {
    layout.sendcounts[proc] = layout.n * layout.cols[proc];
    layout.displs[proc] = disp;
    disp += layout.sendcounts[proc];
  }

  layout.my_first_col = layout.first_col[world_rank];
  layout.my_cols = layout.cols[world_rank];
  return layout;
}

[[nodiscard]] int OwnerOfCol(const Layout &layout, int global_col) {
  for (int proc = 0; proc < layout.world_size; ++proc) {
    const int first = layout.first_col[proc];
    const int count = layout.cols[proc];
    if (global_col >= first && global_col < (first + count)) {
      return proc;
    }
  }
  return -1;
}

[[nodiscard]] bool IsMyCol(const Layout &layout, int global_col) {
  return (global_col >= layout.my_first_col) && (global_col < (layout.my_first_col + layout.my_cols));
}

[[nodiscard]] int LocalCol(const Layout &layout, int global_col) {
  return global_col - layout.my_first_col;
}

struct LocalMatrix {
  int n{};
  int local_cols{};
  std::vector<double> data;  // n * local_cols

  [[nodiscard]] std::size_t Index(int row, int local_col) const {
    return (static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols)) + static_cast<std::size_t>(local_col);
  }

  double &At(int row, int local_col) {
    return data[Index(row, local_col)];
  }

  [[nodiscard]] const double &At(int row, int local_col) const {
    return data[Index(row, local_col)];
  }
};

[[nodiscard]] std::vector<double> PackForScatter(const Layout &layout, const std::vector<double> &aug) {
  const int total = std::accumulate(layout.sendcounts.begin(), layout.sendcounts.end(), 0);
  std::vector<double> packed(static_cast<std::size_t>(total), 0.0);

  const auto m_sz = static_cast<std::size_t>(layout.m);

  for (int proc = 0; proc < layout.world_size; ++proc) {
    const int first = layout.first_col[proc];
    const int count = layout.cols[proc];
    const int off = layout.displs[proc];

    const auto first_sz = static_cast<std::size_t>(first);
    const auto count_sz = static_cast<std::size_t>(count);
    const auto off_sz = static_cast<std::size_t>(off);

    for (int row = 0; row < layout.n; ++row) {
      const auto row_sz = static_cast<std::size_t>(row);
      const auto row_src = row_sz * m_sz;
      const auto row_dst = off_sz + (row_sz * count_sz);

      for (int lc = 0; lc < count; ++lc) {
        const auto lc_sz = static_cast<std::size_t>(lc);
        packed[row_dst + lc_sz] = aug[row_src + first_sz + lc_sz];
      }
    }
  }

  return packed;
}

// swap только в пределах [col_begin..col_end] + RHS (col = n)
void SwapRowsInBand(LocalMatrix &mat, const Layout &layout, int row_a, int row_b, int col_begin, int col_end) {
  if (row_a == row_b) {
    return;
  }

  for (int lc = 0; lc < mat.local_cols; ++lc) {
    const int global_col = layout.my_first_col + lc;
    const bool in_band = (global_col >= col_begin) && (global_col <= col_end);
    const bool is_rhs = (global_col == layout.n);
    if (in_band || is_rhs) {
      std::swap(mat.At(row_a, lc), mat.At(row_b, lc));
    }
  }
}

[[nodiscard]] int FindPivotRow(const LocalMatrix &mat, int local_k_col, int k, int row_end, double eps) {
  int pivot_row = k;
  double best = std::abs(mat.At(k, local_k_col));

  for (int row = k + 1; row <= row_end; ++row) {
    const double v = std::abs(mat.At(row, local_k_col));
    if (v > best) {
      best = v;
      pivot_row = row;
    }
  }

  if (best <= eps) {
    return -1;
  }
  return pivot_row;
}

bool ComputeMultipliers(LocalMatrix &mat, int local_k_col, int k, int row_end, double eps,
                        std::vector<double> &multipliers) {
  const double diag = mat.At(k, local_k_col);
  if (std::abs(diag) <= eps) {
    return false;
  }

  for (int row = k + 1; row <= row_end; ++row) {
    multipliers[static_cast<std::size_t>(row)] = mat.At(row, local_k_col) / diag;
  }

  for (int row = k + 1; row <= row_end; ++row) {
    mat.At(row, local_k_col) = 0.0;
  }

  return true;
}

void ApplyEliminationToLocalCols(LocalMatrix &mat, const Layout &layout, int k, int row_end, int col_end,
                                 const std::vector<double> &multipliers, double eps) {
  for (int lc = 0; lc < mat.local_cols; ++lc) {
    const int global_col = layout.my_first_col + lc;

    const bool update_coeff = (global_col >= (k + 1)) && (global_col <= col_end);
    const bool update_rhs = (global_col == layout.n);
    if (!(update_coeff || update_rhs)) {
      continue;
    }

    const double pivot_val = mat.At(k, lc);

    for (int row = k + 1; row <= row_end; ++row) {
      const double f = multipliers[static_cast<std::size_t>(row)];
      if (std::abs(f) <= eps) {
        continue;
      }
      mat.At(row, lc) -= f * pivot_val;
    }
  }
}

bool ForwardElimination(LocalMatrix &mat, const Layout &layout, int bw, double eps, MPI_Comm comm) {
  const int n = layout.n;
  std::vector<double> multipliers(static_cast<std::size_t>(n), 0.0);

  for (int k = 0; k < n; ++k) {
    const int owner_k = OwnerOfCol(layout, k);
    const int row_end = std::min(n - 1, k + bw);
    const int col_end = std::min(n - 1, k + bw);

    int pivot_row = k;

    if (layout.world_rank == owner_k) {
      const int local_k = LocalCol(layout, k);
      pivot_row = FindPivotRow(mat, local_k, k, row_end, eps);
    }

    MPI_Bcast(&pivot_row, 1, MPI_INT, owner_k, comm);
    if (pivot_row < 0) {
      return false;
    }

    SwapRowsInBand(mat, layout, k, pivot_row, k, col_end);

    std::ranges::fill(multipliers, 0.0);

    int ok = 1;
    if (layout.world_rank == owner_k) {
      const int local_k = LocalCol(layout, k);
      ok = ComputeMultipliers(mat, local_k, k, row_end, eps, multipliers) ? 1 : 0;
    }

    MPI_Bcast(&ok, 1, MPI_INT, owner_k, comm);
    if (ok == 0) {
      return false;
    }

    MPI_Bcast(multipliers.data(), n, MPI_DOUBLE, owner_k, comm);

    ApplyEliminationToLocalCols(mat, layout, k, row_end, col_end, multipliers, eps);
  }

  return true;
}

[[nodiscard]] double ComputeLocalDot(const LocalMatrix &mat, const Layout &layout, const std::vector<double> &x,
                                     int row, int k, int col_end) {
  double sum = 0.0;

  for (int global_col = k + 1; global_col <= col_end; ++global_col) {
    if (!IsMyCol(layout, global_col)) {
      continue;
    }
    const int lc = LocalCol(layout, global_col);
    sum += mat.At(row, lc) * x[static_cast<std::size_t>(global_col)];
  }

  return sum;
}

bool BackSubstitution(const LocalMatrix &mat, const Layout &layout, int bw, double eps, MPI_Comm comm, OutType &x) {
  const int n = layout.n;
  x.assign(static_cast<std::size_t>(n), 0.0);

  const int owner_rhs = OwnerOfCol(layout, n);

  for (int k = n - 1; k >= 0; --k) {
    const int col_end = std::min(n - 1, k + bw);

    const double local_sum = ComputeLocalDot(mat, layout, x, k, k, col_end);

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    double rhs = 0.0;
    if (layout.world_rank == owner_rhs && IsMyCol(layout, n)) {
      rhs = mat.At(k, LocalCol(layout, n));
    }
    MPI_Bcast(&rhs, 1, MPI_DOUBLE, owner_rhs, comm);

    const int owner_diag = OwnerOfCol(layout, k);
    double diag = 0.0;
    if (layout.world_rank == owner_diag && IsMyCol(layout, k)) {
      diag = mat.At(k, LocalCol(layout, k));
    }
    MPI_Bcast(&diag, 1, MPI_DOUBLE, owner_diag, comm);

    if (std::abs(diag) <= eps) {
      return false;
    }

    const double xk = (rhs - global_sum) / diag;
    x[static_cast<std::size_t>(k)] = xk;
  }

  return true;
}

}  // namespace

PeryashkinVGaussVStripMPI::PeryashkinVGaussVStripMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool PeryashkinVGaussVStripMPI::ValidationImpl() {
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

bool PeryashkinVGaussVStripMPI::PreProcessingImpl() {
  return true;
}

bool PeryashkinVGaussVStripMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int n = (world_rank == 0) ? GetInput().n : 0;
  int bw = (world_rank == 0) ? GetInput().bandwidth : 0;
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bw, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const Layout layout = MakeLayout(n, world_size, world_rank);

  LocalMatrix mat;
  mat.n = n;
  mat.local_cols = layout.my_cols;
  mat.data.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(layout.my_cols), 0.0);

  std::vector<double> packed;
  if (world_rank == 0) {
    packed = PackForScatter(layout, GetInput().augmented_matrix);
  }

  MPI_Scatterv((world_rank == 0) ? packed.data() : nullptr, layout.sendcounts.data(), layout.displs.data(), MPI_DOUBLE,
               mat.data.data(), n * layout.my_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  const double eps = std::numeric_limits<double>::epsilon() * kEpsScale;

  if (!ForwardElimination(mat, layout, bw, eps, MPI_COMM_WORLD)) {
    return false;
  }

  OutType x;
  if (!BackSubstitution(mat, layout, bw, eps, MPI_COMM_WORLD, x)) {
    return false;
  }

  GetOutput() = std::move(x);
  return true;
}

bool PeryashkinVGaussVStripMPI::PostProcessingImpl() {
  return GetOutput().size() == static_cast<std::size_t>(GetInput().n);
}

}  // namespace peryashkin_v_gauss_vstrip
