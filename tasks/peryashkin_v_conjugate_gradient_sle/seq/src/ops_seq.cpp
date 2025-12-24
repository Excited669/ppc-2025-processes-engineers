#include "peryashkin_v_conjugate_gradient_sle/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace peryashkin_v_conjugate_gradient_sle {

PeryashkinVConjGradSleSEQ::PeryashkinVConjGradSleSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool PeryashkinVConjGradSleSEQ::ValidationImpl() {
  const auto [n, variant] = GetInput();
  return (n > 0) && (variant >= 0 && variant <= 2) && GetOutput().empty();
}

bool PeryashkinVConjGradSleSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

namespace {

void ApplyA(const std::vector<double> &x, int variant, std::vector<double> *y) {
  const std::size_t n = x.size();
  y->assign(n, 0.0);

  if (variant == 1) {
    for (std::size_t i = 0; i < n; ++i) {
      (*y)[i] = 5.0 * x[i];
    }
    return;
  }

  if (variant == 0) {
    for (std::size_t i = 0; i < n; ++i) {
      double v = 4.0 * x[i];
      if (i > 0) {
        v += x[i - 1];
      }
      if (i + 1 < n) {
        v += x[i + 1];
      }
      (*y)[i] = v;
    }
    return;
  }

  // variant == 2
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t j = n - 1 - i;
    double v = 3.0 * x[i];
    if (j != i) {
      v -= x[j];
    }
    (*y)[i] = v;
  }
}

double Dot(const std::vector<double> &a, const std::vector<double> &b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    s += a[i] * b[i];
  }
  return s;
}

void ConjugateGradient(int n, int variant, const std::vector<double> &b, std::vector<double> *x) {
  const double eps = 1e-7;
  const int max_iters = std::max(2000, 2 * n);

  std::vector<double> r = b;  // x starts with zeros -> r=b-Ax=b
  std::vector<double> p = r;
  std::vector<double> Ap(static_cast<std::size_t>(n), 0.0);

  double rr = Dot(r, r);

  for (int it = 0; it < max_iters; ++it) {
    if (std::sqrt(rr) < eps) {
      break;
    }

    ApplyA(p, variant, &Ap);

    const double pAp = Dot(p, Ap);
    if (std::fabs(pAp) < 1e-15) {
      break;
    }

    const double alpha = rr / pAp;

    for (int i = 0; i < n; ++i) {
      (*x)[static_cast<std::size_t>(i)] += alpha * p[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < n; ++i) {
      r[static_cast<std::size_t>(i)] -= alpha * Ap[static_cast<std::size_t>(i)];
    }

    const double rr_new = Dot(r, r);
    const double beta = rr_new / rr;

    for (int i = 0; i < n; ++i) {
      const std::size_t idx = static_cast<std::size_t>(i);
      p[idx] = r[idx] + beta * p[idx];
    }

    rr = rr_new;
  }
}

}  // namespace

bool PeryashkinVConjGradSleSEQ::RunImpl() {
  const auto [n, variant] = GetInput();
  if (n <= 0) {
    return false;
  }

  std::vector<double> b(static_cast<std::size_t>(n), 1.0);
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);

  ConjugateGradient(n, variant, b, &x);

  GetOutput() = std::move(x);
  return true;
}

bool PeryashkinVConjGradSleSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace peryashkin_v_conjugate_gradient_sle
