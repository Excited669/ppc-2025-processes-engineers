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
    ok = ((n > 0) && (variant >= 0 && variant <= 2) && GetOutput().empty()) ? 1 : 0;
  }

  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok == 1;
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
    const int add = (proc < extra) ? 1 : 0;
    (*counts)[static_cast<std::size_t>(proc)] = base + add;
    (*displs)[static_cast<std::size_t>(proc)] = disp;
    disp += base + add;
  }
}

double DotLocal(const std::vector<double> &a, const std::vector<double> &b) {
  double sum = 0.0;
  const std::size_t sz = a.size();
  for (std::size_t i = 0; i < sz; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

double DotAllreduce(const std::vector<double> &a, const std::vector<double> &b) {
  const double local = DotLocal(a, b);
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global;
}

std::pair<double, double> ExchangeTridiagHalos(int rank, int world_size, const std::vector<double> &x_local) {
  const int left_rank = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
  const int right_rank = (rank + 1 < world_size) ? (rank + 1) : MPI_PROC_NULL;

  const int local_rows = static_cast<int>(x_local.size());
  const double send_left = (local_rows > 0) ? x_local.front() : 0.0;
  const double send_right = (local_rows > 0) ? x_local.back() : 0.0;

  double recv_left = 0.0;
  double recv_right = 0.0;

  MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, left_rank, 10, &recv_right, 1, MPI_DOUBLE, right_rank, 10, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, right_rank, 11, &recv_left, 1, MPI_DOUBLE, left_rank, 11, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  return {recv_left, recv_right};
}

void ApplyTridiagLocal(int n, int local_start, const std::vector<double> &x_local, double halo_left, double halo_right,
                       std::vector<double> *y_local) {
  const int local_rows = static_cast<int>(x_local.size());
  y_local->assign(static_cast<std::size_t>(local_rows), 0.0);

  for (int i = 0; i < local_rows; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    const int global_i = local_start + i;

    double val = 4.0 * x_local[idx];

    if (global_i > 0) {
      if (i > 0) {
        val += x_local[idx - 1];
      } else {
        val += halo_left;
      }
    }

    if (global_i + 1 < n) {
      if (i + 1 < local_rows) {
        val += x_local[idx + 1];
      } else {
        val += halo_right;
      }
    }

    (*y_local)[idx] = val;
  }
}

void ApplyOperatorLocal(int n, int variant, int rank, int world_size, int local_start, const std::vector<int> &counts,
                        const std::vector<int> &displs, const std::vector<double> &x_local,
                        std::vector<double> *y_local) {
  const int local_rows = static_cast<int>(x_local.size());
  y_local->assign(static_cast<std::size_t>(local_rows), 0.0);

  if (variant == 1) {
    for (int i = 0; i < local_rows; ++i) {
      (*y_local)[static_cast<std::size_t>(i)] = 5.0 * x_local[static_cast<std::size_t>(i)];
    }
    return;
  }

  if (variant == 0) {
    const auto halos = ExchangeTridiagHalos(rank, world_size, x_local);
    ApplyTridiagLocal(n, local_start, x_local, halos.first, halos.second, y_local);
    return;
  }

  // variant == 2 (centrosymmetric): needs x[j], so we gather full vector
  std::vector<double> x_full(static_cast<std::size_t>(n), 0.0);
  MPI_Allgatherv(x_local.data(), local_rows, MPI_DOUBLE, x_full.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);

  for (int i = 0; i < local_rows; ++i) {
    const int global_i = local_start + i;
    const int mirror_i = (n - 1) - global_i;

    double val = 3.0 * x_full[static_cast<std::size_t>(global_i)];
    if (mirror_i != global_i) {
      val -= x_full[static_cast<std::size_t>(mirror_i)];
    }
    (*y_local)[static_cast<std::size_t>(i)] = val;
  }
}

void ConjugateGradientLocal(int n, int variant, int rank, int world_size, int local_start,
                            const std::vector<int> &counts, const std::vector<int> &displs,
                            const std::vector<double> &b_local, std::vector<double> *x_local) {
  constexpr double kEps = 1e-7;
  constexpr int kMaxIters = 2000;

  const int local_rows = static_cast<int>(b_local.size());
  std::vector<double> r_local = b_local;
  std::vector<double> p_local = r_local;
  std::vector<double> ap_local(static_cast<std::size_t>(local_rows), 0.0);

  // x0 = 0 => A*x0 = 0, so r0 = b
  x_local->assign(static_cast<std::size_t>(local_rows), 0.0);

  double rr = DotAllreduce(r_local, r_local);

  for (int it = 0; it < kMaxIters; ++it) {
    if (std::sqrt(rr) < kEps) {
      break;
    }

    ApplyOperatorLocal(n, variant, rank, world_size, local_start, counts, displs, p_local, &ap_local);

    const double p_ap = DotAllreduce(p_local, ap_local);
    if (std::fabs(p_ap) < 1e-15) {
      break;
    }

    const double alpha = rr / p_ap;

    for (int i = 0; i < local_rows; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      (*x_local)[idx] += alpha * p_local[idx];
      r_local[idx] -= alpha * ap_local[idx];
    }

    const double rr_new = DotAllreduce(r_local, r_local);
    const double beta = rr_new / rr;

    for (int i = 0; i < local_rows; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      p_local[idx] = r_local[idx] + (beta * p_local[idx]);
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

  std::vector<double> b_full;
  if (rank == 0) {
    b_full.assign(static_cast<std::size_t>(n), 1.0);
  }

  std::vector<double> b_local(static_cast<std::size_t>(local_rows), 0.0);
  MPI_Scatterv(b_full.data(), counts.data(), displs.data(), MPI_DOUBLE, b_local.data(), local_rows, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  std::vector<double> x_local;
  ConjugateGradientLocal(n, variant, rank, world_size, local_start, counts, displs, b_local, &x_local);

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
