#include "peryashkin_v_conjugate_gradient_sle/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "peryashkin_v_conjugate_gradient_sle/common/include/common.hpp"

namespace peryashkin_v_conjugate_gradient_sle {

PeryashkinVConjGradSleSEQ::PeryashkinVConjGradSleSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool PeryashkinVConjGradSleSEQ::ValidationImpl() {
  const int n = GetInput().first;
  const int variant = GetInput().second;
  return (n > 0) && (variant >= 0 && variant <= 2) && GetOutput().empty();
}

bool PeryashkinVConjGradSleSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

double Dot(const std::vector<double> &a, const std::vector<double> &b) {
  double sum = 0.0;
  const std::size_t sz = a.size();
  for (std::size_t i = 0; i < sz; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

void ApplyA(const std::vector<double> &x, int variant, std::vector<double> *y) {
  const int n = static_cast<int>(x.size());
  y->assign(static_cast<std::size_t>(n), 0.0);

  if (variant == 1) {
    for (int i = 0; i < n; ++i) {
      (*y)[static_cast<std::size_t>(i)] = 5.0 * x[static_cast<std::size_t>(i)];
    }
    return;
  }

  if (variant == 0) {
    for (int i = 0; i < n; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      double val = 4.0 * x[idx];

      if (i > 0) {
        val += x[idx - 1];
      }
      if (i + 1 < n) {
        val += x[idx + 1];
      }

      (*y)[idx] = val;
    }
    return;
  }

  // variant == 2
  for (int i = 0; i < n; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    const int mirror_i = (n - 1) - i;
    const auto mirror = static_cast<std::size_t>(mirror_i);

    double val = 3.0 * x[idx];
    if (mirror_i != i) {
      val -= x[mirror];
    }
    (*y)[idx] = val;
  }
}

void ConjugateGradient(int n, int variant, const std::vector<double> &b, std::vector<double> *x) {
  constexpr double kEps = 1e-7;
  constexpr int kMaxIters = 2000;

  x->assign(static_cast<std::size_t>(n), 0.0);

  std::vector<double> r_vec(static_cast<std::size_t>(n), 0.0);
  std::vector<double> p_dir(static_cast<std::size_t>(n), 0.0);
  std::vector<double> ap_vec(static_cast<std::size_t>(n), 0.0);

  ApplyA(*x, variant, &ap_vec);
  for (int i = 0; i < n; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    r_vec[idx] = b[idx] - ap_vec[idx];
    p_dir[idx] = r_vec[idx];
  }

  double rr = Dot(r_vec, r_vec);

  for (int it = 0; it < kMaxIters; ++it) {
    if (std::sqrt(rr) < kEps) {
      break;
    }

    ApplyA(p_dir, variant, &ap_vec);

    const double p_ap = Dot(p_dir, ap_vec);
    if (std::fabs(p_ap) < 1e-15) {
      break;
    }

    const double alpha = rr / p_ap;

    for (int i = 0; i < n; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      (*x)[idx] += alpha * p_dir[idx];
      r_vec[idx] -= alpha * ap_vec[idx];
    }

    const double rr_new = Dot(r_vec, r_vec);
    const double beta = rr_new / rr;

    for (int i = 0; i < n; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      p_dir[idx] = r_vec[idx] + (beta * p_dir[idx]);
    }

    rr = rr_new;
  }
}

}  // namespace

bool PeryashkinVConjGradSleSEQ::RunImpl() {
  const int n = GetInput().first;
  const int variant = GetInput().second;

  if (n <= 0) {
    return false;
  }

  std::vector<double> b(static_cast<std::size_t>(n), 1.0);
  std::vector<double> x;

  ConjugateGradient(n, variant, b, &x);

  GetOutput() = std::move(x);
  return true;
}

bool PeryashkinVConjGradSleSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace peryashkin_v_conjugate_gradient_sle
