#include "peryashkin_v_conjugate_gradient_sle/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "peryashkin_v_conjugate_gradient_sle/common/include/common.hpp"

namespace peryashkin_v_conjugate_gradient_sle {

PeryashkinVConjGradSleMPI::PeryashkinVConjGradSleMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool PeryashkinVConjGradSleMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ok = 0;
  if (rank == 0) {
    const int n = GetInput().first;
    const int variant = GetInput().second;
    const bool is_valid = (n > 0) && (variant >= 0) && (variant <= 2) && GetOutput().empty();
    ok = is_valid ? 1 : 0;
  }

  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok != 0;
}

bool PeryashkinVConjGradSleMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().clear();
  }
  return true;
}

namespace {

void CalcCountsDispls(int n, int world_size, std::vector<int> *counts, std::vector<int> *displs) {
  counts->assign(static_cast<std::size_t>(world_size), 0);
  displs->assign(static_cast<std::size_t>(world_size), 0);

  const int base = n / world_size;
  const int extra = n % world_size;

  int disp = 0;
  for (int proc = 0; proc < world_size; ++proc) {
    const int cnt = base + ((proc < extra) ? 1 : 0);
    (*counts)[static_cast<std::size_t>(proc)] = cnt;
    (*displs)[static_cast<std::size_t>(proc)] = disp;
    disp += cnt;
  }
}

double LocalDot(const std::vector<double> &a, const std::vector<double> &b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

void ApplyADiagonalLocal(const std::vector<double> &x_local, std::vector<double> *y_local) {
  y_local->assign(x_local.size(), 0.0);
  for (std::size_t i = 0; i < x_local.size(); ++i) {
    (*y_local)[i] = 5.0 * x_local[i];
  }
}

void ApplyATridiagLocal(int n, int rank, int world_size, int local_start, const std::vector<double> &x_local,
                        std::vector<double> *y_local) {
  const int local_rows = static_cast<int>(x_local.size());
  y_local->assign(x_local.size(), 0.0);

  const int left_rank = (rank - 1 >= 0) ? (rank - 1) : MPI_PROC_NULL;
  const int right_rank = (rank + 1 < world_size) ? (rank + 1) : MPI_PROC_NULL;

  const double send_left = (local_rows > 0) ? x_local.front() : 0.0;
  const double send_right = (local_rows > 0) ? x_local.back() : 0.0;

  double recv_left = 0.0;
  double recv_right = 0.0;

  MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, left_rank, 0, &recv_right, 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, right_rank, 1, &recv_left, 1, MPI_DOUBLE, left_rank, 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  for (int i = 0; i < local_rows; ++i) {
    const int global_i = local_start + i;
    const auto idx = static_cast<std::size_t>(i);

    double val = 4.0 * x_local[idx];

    if (global_i > 0) {
      if (i > 0) {
        val += x_local[idx - 1];
      } else {
        val += recv_left;
      }
    }

    if (global_i + 1 < n) {
      if (i + 1 < local_rows) {
        val += x_local[idx + 1];
      } else {
        val += recv_right;
      }
    }

    (*y_local)[idx] = val;
  }
}

void ApplyACentrosymLocal(int n, int local_start, const std::vector<int> &counts, const std::vector<int> &displs,
                          const std::vector<double> &x_local, std::vector<double> *y_local) {
  const int local_rows = static_cast<int>(x_local.size());
  y_local->assign(x_local.size(), 0.0);

  std::vector<double> x_full(static_cast<std::size_t>(n), 0.0);
  MPI_Allgatherv(x_local.data(), local_rows, MPI_DOUBLE, x_full.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);

  for (int i = 0; i < local_rows; ++i) {
    const int global_i = local_start + i;
    const int j = (n - 1) - global_i;

    const auto idx = static_cast<std::size_t>(i);
    double val = 3.0 * x_full[static_cast<std::size_t>(global_i)];
    if (j != global_i) {
      val -= x_full[static_cast<std::size_t>(j)];
    }
    (*y_local)[idx] = val;
  }
}

void ApplyALocal(int n, int variant, int rank, int world_size, int local_start, const std::vector<int> &counts,
                 const std::vector<int> &displs, const std::vector<double> &x_local, std::vector<double> *y_local) {
  if (variant == 1) {
    ApplyADiagonalLocal(x_local, y_local);
    return;
  }
  if (variant == 0) {
    ApplyATridiagLocal(n, rank, world_size, local_start, x_local, y_local);
    return;
  }
  ApplyACentrosymLocal(n, local_start, counts, displs, x_local, y_local);
}

void ConjugateGradientMpi(int n, int variant, int rank, int world_size, int local_start, const std::vector<int> &counts,
                          const std::vector<int> &displs, const std::vector<double> &b_local,
                          std::vector<double> *x_local) {
  constexpr double eps = 1e-7;
  constexpr int max_iters = 2000;

  std::vector<double> r_local = b_local;
  std::vector<double> p_local = r_local;
  std::vector<double> ap_local(static_cast<std::size_t>(b_local.size()), 0.0);

  double rr_local = LocalDot(r_local, r_local);
  double rr = 0.0;
  MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  for (int it = 0; it < max_iters; ++it) {
    if (std::sqrt(rr) < eps) {
      break;
    }

    ApplyALocal(n, variant, rank, world_size, local_start, counts, displs, p_local, &ap_local);

    const double p_ap_local = LocalDot(p_local, ap_local);
    double p_ap = 0.0;
    MPI_Allreduce(&p_ap_local, &p_ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (std::fabs(p_ap) < 1e-15) {
      break;
    }

    const double alpha = rr / p_ap;

    for (std::size_t i = 0; i < r_local.size(); ++i) {
      (*x_local)[i] += alpha * p_local[i];
      r_local[i] -= alpha * ap_local[i];
    }

    const double rr_new_local = LocalDot(r_local, r_local);
    double rr_new = 0.0;
    MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const double beta = rr_new / rr;

    for (std::size_t i = 0; i < p_local.size(); ++i) {
      p_local[i] = r_local[i] + (beta * p_local[i]);
    }

    rr = rr_new;
  }
}

}  // namespace

bool PeryashkinVConjGradSleMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int n = 0;
  int variant = 0;
  if (rank == 0) {
    n = GetInput().first;
    variant = GetInput().second;
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&variant, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 0) {
    return false;
  }

  std::vector<int> counts;
  std::vector<int> displs;
  CalcCountsDispls(n, world_size, &counts, &displs);

  const int local_rows = counts[static_cast<std::size_t>(rank)];
  const int local_start = displs[static_cast<std::size_t>(rank)];

  std::vector<double> b_local(static_cast<std::size_t>(local_rows), 1.0);
  std::vector<double> x_local(static_cast<std::size_t>(local_rows), 0.0);

  ConjugateGradientMpi(n, variant, rank, world_size, local_start, counts, displs, b_local, &x_local);

  std::vector<double> x_full;
  if (rank == 0) {
    x_full.assign(static_cast<std::size_t>(n), 0.0);
  }

  MPI_Gatherv(x_local.data(), local_rows, MPI_DOUBLE, x_full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = std::move(x_full);
  }

  return true;
}

bool PeryashkinVConjGradSleMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetOutput().empty();
  }
  return true;
}

}  // namespace peryashkin_v_conjugate_gradient_sle
