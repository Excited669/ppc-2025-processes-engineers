#include "peryashkin_v_conjugate_gradient_sle/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

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
    const auto [n, variant] = GetInput();
    ok = (n > 0) && (variant >= 0 && variant <= 2) && GetOutput().empty();
    ok = ok ? 1 : 0;
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

void CalcRowDist(int n, int world_size, std::vector<int> *counts, std::vector<int> *displs) {
  counts->assign(world_size, 0);
  displs->assign(world_size, 0);

  const int base = n / world_size;
  const int extra = n % world_size;

  int disp = 0;
  for (int p = 0; p < world_size; ++p) {
    const int c = base + (p < extra ? 1 : 0);
    (*counts)[p] = c;
    (*displs)[p] = disp;
    disp += c;
  }
}

double LocalDot(const std::vector<double> &a, const std::vector<double> &b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    s += a[i] * b[i];
  }
  return s;
}

void ApplyA_local(int n, int variant, int rank, int world_size, int local_start, const std::vector<int> &counts,
                  const std::vector<int> &displs, const std::vector<double> &x_local, std::vector<double> *y_local) {
  const int local_rows = static_cast<int>(x_local.size());
  y_local->assign(static_cast<std::size_t>(local_rows), 0.0);

  if (variant == 1) {
    for (int i = 0; i < local_rows; ++i) {
      (*y_local)[static_cast<std::size_t>(i)] = 5.0 * x_local[static_cast<std::size_t>(i)];
    }
    return;
  }

  if (variant == 0) {
    // halo exchange: needs neighbors for first/last local element
    double left_ghost = 0.0;
    double right_ghost = 0.0;

    const int left_rank = (rank - 1 >= 0) ? rank - 1 : MPI_PROC_NULL;
    const int right_rank = (rank + 1 < world_size) ? rank + 1 : MPI_PROC_NULL;

    const double send_left = (local_rows > 0) ? x_local.front() : 0.0;
    const double send_right = (local_rows > 0) ? x_local.back() : 0.0;

    // receive right ghost from right neighbor, send left boundary to left neighbor
    MPI_Sendrecv(&send_left, 1, MPI_DOUBLE, left_rank, 10, &right_ghost, 1, MPI_DOUBLE, right_rank, 10, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    // receive left ghost from left neighbor, send right boundary to right neighbor
    MPI_Sendrecv(&send_right, 1, MPI_DOUBLE, right_rank, 11, &left_ghost, 1, MPI_DOUBLE, left_rank, 11, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    for (int i = 0; i < local_rows; ++i) {
      const int global_i = local_start + i;

      double v = 4.0 * x_local[static_cast<std::size_t>(i)];

      // left neighbor
      if (global_i > 0) {
        if (i > 0) {
          v += x_local[static_cast<std::size_t>(i - 1)];
        } else {
          v += left_ghost;
        }
      }

      // right neighbor
      if (global_i + 1 < n) {
        if (i + 1 < local_rows) {
          v += x_local[static_cast<std::size_t>(i + 1)];
        } else {
          v += right_ghost;
        }
      }

      (*y_local)[static_cast<std::size_t>(i)] = v;
    }
    return;
  }

  // variant == 2: need x[n-1-i] -> allgather local x into global x
  std::vector<double> x_global(static_cast<std::size_t>(n), 0.0);
  MPI_Allgatherv(x_local.data(), local_rows, MPI_DOUBLE, x_global.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);

  for (int i = 0; i < local_rows; ++i) {
    const int global_i = local_start + i;
    const int j = n - 1 - global_i;

    double v = 3.0 * x_local[static_cast<std::size_t>(i)];
    if (j != global_i) {
      v -= x_global[static_cast<std::size_t>(j)];
    }

    (*y_local)[static_cast<std::size_t>(i)] = v;
  }
}

void ConjugateGradientMPI(int n, int variant, int rank, int world_size, int local_start, const std::vector<int> &counts,
                          const std::vector<int> &displs, const std::vector<double> &b_local,
                          std::vector<double> *x_local) {
  const double eps = 1e-7;
  const int max_iters = std::max(2000, 2 * n);

  const int local_rows = static_cast<int>(b_local.size());

  std::vector<double> r_local = b_local;  // x=0 -> r=b
  std::vector<double> p_local = r_local;
  std::vector<double> Ap_local(static_cast<std::size_t>(local_rows), 0.0);

  double rr_local = LocalDot(r_local, r_local);
  double rr = 0.0;
  MPI_Allreduce(&rr_local, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  for (int it = 0; it < max_iters; ++it) {
    if (std::sqrt(rr) < eps) {
      break;
    }

    ApplyA_local(n, variant, rank, world_size, local_start, counts, displs, p_local, &Ap_local);

    const double pAp_local = LocalDot(p_local, Ap_local);
    double pAp = 0.0;
    MPI_Allreduce(&pAp_local, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (std::fabs(pAp) < 1e-15) {
      break;
    }

    const double alpha = rr / pAp;

    for (int i = 0; i < local_rows; ++i) {
      (*x_local)[static_cast<std::size_t>(i)] += alpha * p_local[static_cast<std::size_t>(i)];
      r_local[static_cast<std::size_t>(i)] -= alpha * Ap_local[static_cast<std::size_t>(i)];
    }

    const double rr_new_local = LocalDot(r_local, r_local);
    double rr_new = 0.0;
    MPI_Allreduce(&rr_new_local, &rr_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const double beta = rr_new / rr;

    for (int i = 0; i < local_rows; ++i) {
      const std::size_t idx = static_cast<std::size_t>(i);
      p_local[idx] = r_local[idx] + beta * p_local[idx];
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
  CalcRowDist(n, world_size, &counts, &displs);

  const int local_rows = counts[rank];
  const int local_start = displs[rank];

  // b = ones (root creates full, then scatter)
  std::vector<double> b_full;
  if (rank == 0) {
    b_full.assign(static_cast<std::size_t>(n), 1.0);
  }

  std::vector<double> b_local(static_cast<std::size_t>(local_rows), 0.0);
  double *sendbuf = (rank == 0) ? b_full.data() : nullptr;

  MPI_Scatterv(sendbuf, counts.data(), displs.data(), MPI_DOUBLE, b_local.data(), local_rows, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  std::vector<double> x_local(static_cast<std::size_t>(local_rows), 0.0);

  ConjugateGradientMPI(n, variant, rank, world_size, local_start, counts, displs, b_local, &x_local);

  // gather x to root
  std::vector<double> x_full;
  if (rank == 0) {
    x_full.assign(static_cast<std::size_t>(n), 0.0);
  }

  double *recvbuf = (rank == 0) ? x_full.data() : nullptr;
  MPI_Gatherv(x_local.data(), local_rows, MPI_DOUBLE, recvbuf, counts.data(), displs.data(), MPI_DOUBLE, 0,
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
