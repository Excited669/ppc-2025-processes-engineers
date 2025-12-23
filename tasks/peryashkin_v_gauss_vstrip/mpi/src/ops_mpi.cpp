#include "peryashkin_v_gauss_vstrip/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace peryashkin_v_gauss_vstrip {

namespace {

constexpr double kEpsScale = 256.0;

struct Layout {
  int n{};
  int m{};  // n+1
  int size{};
  int rank{};
  int my_first{};
  int my_cols{};
  std::vector<int> first;
  std::vector<int> cols;
  std::vector<int> sendcounts;
  std::vector<int> displs;
};

Layout MakeLayout(int n, int size, int rank) {
  Layout L;
  L.n = n;
  L.m = n + 1;
  L.size = size;
  L.rank = rank;

  L.first.assign(size, 0);
  L.cols.assign(size, 0);
  L.sendcounts.assign(size, 0);
  L.displs.assign(size, 0);

  const int base = L.m / size;
  const int rem = L.m % size;

  int cur = 0;
  for (int r = 0; r < size; ++r) {
    const int c = base + (r < rem ? 1 : 0);
    L.first[r] = cur;
    L.cols[r] = c;
    cur += c;
  }

  int disp = 0;
  for (int r = 0; r < size; ++r) {
    L.sendcounts[r] = L.n * L.cols[r];
    L.displs[r] = disp;
    disp += L.sendcounts[r];
  }

  L.my_first = L.first[rank];
  L.my_cols = L.cols[rank];
  return L;
}

int OwnerOfCol(const Layout& L, int j) {
  for (int r = 0; r < L.size; ++r) {
    const int f = L.first[r];
    const int c = L.cols[r];
    if (j >= f && j < f + c) return r;
  }
  return -1;
}

bool IsMyCol(const Layout& L, int j) {
  return (j >= L.my_first) && (j < L.my_first + L.my_cols);
}

int LocalCol(const Layout& L, int j) { return j - L.my_first; }

struct LocalMat {
  int n{};
  int local_cols{};
  std::vector<double> a;  // n * local_cols

  double& at(int row, int lcol) {
    return a[static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols) + static_cast<std::size_t>(lcol)];
  }
  const double& at(int row, int lcol) const {
    return a[static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols) + static_cast<std::size_t>(lcol)];
  }
};

std::vector<double> PackForScatter(const Layout& L, const std::vector<double>& aug) {
  const int total = std::accumulate(L.sendcounts.begin(), L.sendcounts.end(), 0);
  std::vector<double> packed(static_cast<std::size_t>(total), 0.0);

  for (int r = 0; r < L.size; ++r) {
    const int f = L.first[r];
    const int c = L.cols[r];
    const int off = L.displs[r];

    for (int i = 0; i < L.n; ++i) {
      const int row_src = i * L.m;
      const int row_dst = off + i * c;
      for (int lc = 0; lc < c; ++lc) {
        packed[static_cast<std::size_t>(row_dst + lc)] =
            aug[static_cast<std::size_t>(row_src + (f + lc))];
      }
    }
  }

  return packed;
}

void SwapRows(LocalMat& M, int r1, int r2) {
  if (r1 == r2) return;
  for (int lc = 0; lc < M.local_cols; ++lc) {
    std::swap(M.at(r1, lc), M.at(r2, lc));
  }
}

}  // namespace

PeryashkinVGaussVStripMPI::PeryashkinVGaussVStripMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool PeryashkinVGaussVStripMPI::ValidationImpl() {
  const auto& in = GetInput();
  if (in.n <= 0) return false;
  if (in.bandwidth < 0 || in.bandwidth >= in.n) return false;

  const std::size_t need =
      static_cast<std::size_t>(in.n) * static_cast<std::size_t>(in.n + 1);
  if (in.augmented_matrix.size() != need) return false;

  return GetOutput().empty();
}

bool PeryashkinVGaussVStripMPI::PreProcessingImpl() { return true; }

bool PeryashkinVGaussVStripMPI::RunImpl() {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = (rank == 0 ? GetInput().n : 0);
  int bw = (rank == 0 ? GetInput().bandwidth : 0);
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bw, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const Layout L = MakeLayout(n, size, rank);

  LocalMat M;
  M.n = n;
  M.local_cols = L.my_cols;
  M.a.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(L.my_cols), 0.0);

  std::vector<double> packed;
  if (rank == 0) packed = PackForScatter(L, GetInput().augmented_matrix);

  MPI_Scatterv(rank == 0 ? packed.data() : nullptr,
               L.sendcounts.data(), L.displs.data(), MPI_DOUBLE,
               M.a.data(), n * L.my_cols, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

  const double eps = std::numeric_limits<double>::epsilon() * kEpsScale;

  std::vector<double> mult(static_cast<std::size_t>(n), 0.0);

  // forward elimination (band limited)
  for (int k = 0; k < n; ++k) {
    const int owner_k = OwnerOfCol(L, k);

    int pivot_row = k;
    if (rank == owner_k) {
      const int lk = LocalCol(L, k);
      const int row_end = std::min(n - 1, k + bw);

      double best = std::abs(M.at(k, lk));
      pivot_row = k;
      for (int r = k + 1; r <= row_end; ++r) {
        const double v = std::abs(M.at(r, lk));
        if (v > best) {
          best = v;
          pivot_row = r;
        }
      }
      if (best <= eps) pivot_row = -1;
    }

    MPI_Bcast(&pivot_row, 1, MPI_INT, owner_k, MPI_COMM_WORLD);
    if (pivot_row < 0) return false;

    SwapRows(M, k, pivot_row);

    std::fill(mult.begin(), mult.end(), 0.0);
    const int row_end = std::min(n - 1, k + bw);

    if (rank == owner_k) {
      const int lk = LocalCol(L, k);
      const double diag = M.at(k, lk);
      if (std::abs(diag) <= eps) return false;

      for (int i = k + 1; i <= row_end; ++i) {
        mult[static_cast<std::size_t>(i)] = M.at(i, lk) / diag;
      }
      for (int i = k + 1; i <= row_end; ++i) {
        M.at(i, lk) = 0.0;
      }
    }

    MPI_Bcast(mult.data(), n, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);

    const int col_end = std::min(n - 1, k + bw);

    auto update_col = [&](int j) {
      if (!IsMyCol(L, j)) return;
      const int lj = LocalCol(L, j);
      const double pivot_val = M.at(k, lj);
      for (int i = k + 1; i <= row_end; ++i) {
        const double f = mult[static_cast<std::size_t>(i)];
        if (std::abs(f) <= eps) continue;
        M.at(i, lj) -= f * pivot_val;
      }
    };

    for (int j = k + 1; j <= col_end; ++j) update_col(j);
    update_col(n);  // RHS
  }

  // backward substitution
  OutType x(static_cast<std::size_t>(n), 0.0);
  const int owner_b = OwnerOfCol(L, n);

  for (int k = n - 1; k >= 0; --k) {
    const int col_end = std::min(n - 1, k + bw);

    double part = 0.0;
    for (int j = k + 1; j <= col_end; ++j) {
      if (!IsMyCol(L, j)) continue;
      part += M.at(k, LocalCol(L, j)) * x[static_cast<std::size_t>(j)];
    }

    double sum = 0.0;
    MPI_Allreduce(&part, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double bk = 0.0;
    if (rank == owner_b) bk = M.at(k, LocalCol(L, n));
    MPI_Bcast(&bk, 1, MPI_DOUBLE, owner_b, MPI_COMM_WORLD);

    const int owner_k = OwnerOfCol(L, k);
    double xk = 0.0;
    if (rank == owner_k) {
      const double diag = M.at(k, LocalCol(L, k));
      if (std::abs(diag) <= eps) {
        xk = std::numeric_limits<double>::quiet_NaN();
      } else {
        xk = (bk - sum) / diag;
      }
    }
    MPI_Bcast(&xk, 1, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);
    if (!std::isfinite(xk)) return false;

    x[static_cast<std::size_t>(k)] = xk;
  }

  GetOutput() = std::move(x);
  return true;
}

bool PeryashkinVGaussVStripMPI::PostProcessingImpl() {
  return GetOutput().size() == static_cast<std::size_t>(GetInput().n);
}

}  // namespace peryashkin_v_gauss_vstrip
