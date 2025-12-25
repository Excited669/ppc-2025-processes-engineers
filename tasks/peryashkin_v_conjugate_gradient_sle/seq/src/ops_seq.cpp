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
  return (n > 0) && (variant >= 0) && (variant <= 2) && GetOutput().empty();
}

bool PeryashkinVConjGradSleSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

void ApplyA(const std::vector<double> &x, int variant, std::vector<double> *y) {
  const auto n = static_cast<int>(x.size());
  y->assign(x.size(), 0.0);

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
    const int j = (n - 1) - i;
    double val = 3.0 * x[idx];
    if (j != i) {
      val -= x[static_cast<std::size_t>(j)];
    }
    (*y)[idx] = val;
  }
}

double Dot(const std::vector<double> &a, const std::vector<double> &b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

void ConjugateGradientSeq(int n, int variant, const std::vector<double> &b, std::vector<double> *x) {
  constexpr double eps = 1e-7;
  constexpr int max_iters = 2000;

  std::vector<double> r(static_cast<std::size_t>(n), 0.0);
  std::vector<double> p(static_cast<std::size_t>(n), 0.0);
  std::vector<double> ap(static_cast<std::size_t>(n), 0.0);

  ApplyA(*x, variant, &ap);
  for (int i = 0; i < n; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    r[idx] = b[idx] - ap[idx];
    p[idx] = r[idx];
  }

  double rr = Dot(r, r);

  for (int it = 0; it < max_iters; ++it) {
    if (std::sqrt(rr) < eps) {
      break;
    }

    ApplyA(p, variant, &ap);

    const double p_ap = Dot(p, ap);
    if (std::fabs(p_ap) < 1e-15) {
      break;
    }

    const double alpha = rr / p_ap;

    for (int i = 0; i < n; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      (*x)[idx] += alpha * p[idx];
      r[idx] -= alpha * ap[idx];
    }

    const double rr_new = Dot(r, r);
    const double beta = rr_new / rr;

    for (int i = 0; i < n; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      p[idx] = r[idx] + (beta * p[idx]);
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
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);

  ConjugateGradientSeq(n, variant, b, &x);

  GetOutput() = std::move(x);
  return true;
}

bool PeryashkinVConjGradSleSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace peryashkin_v_conjugate_gradient_sle
