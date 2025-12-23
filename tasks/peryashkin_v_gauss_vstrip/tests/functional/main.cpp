#include <gtest/gtest.h>

#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "peryashkin_v_gauss_vstrip/common/include/common.hpp"
#include "peryashkin_v_gauss_vstrip/mpi/include/ops_mpi.hpp"
#include "peryashkin_v_gauss_vstrip/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace peryashkin_v_gauss_vstrip {

namespace {

std::string Sanitize(std::string s) {
  for (char& ch : s) {
    const unsigned char u = static_cast<unsigned char>(ch);
    if (std::isalnum(u) == 0 && ch != '_') ch = '_';
  }
  return s;
}

GaussBandInput LoadSystem(const std::string& filename) {
  GaussBandInput in;
  const std::string abs =
      ppc::util::GetAbsoluteTaskPath(PPC_ID_peryashkin_v_gauss_vstrip, filename);

  std::ifstream fin(abs);
  if (!fin.is_open()) throw std::runtime_error("Cannot open: " + abs);

  fin >> in.n >> in.bandwidth;
  if (in.n <= 0) throw std::runtime_error("Bad n in: " + abs);

  const std::size_t total =
      static_cast<std::size_t>(in.n) * static_cast<std::size_t>(in.n + 1);
  in.augmented_matrix.assign(total, 0.0);

  for (std::size_t i = 0; i < total; ++i) {
    if (!(fin >> in.augmented_matrix[i])) {
      throw std::runtime_error("Bad matrix data in: " + abs);
    }
  }
  return in;
}

bool SolveDenseRef(std::vector<double> a, int n, std::vector<double>& x) {
  const double eps = 1e-12;
  auto id = [&](int r, int c) -> std::size_t {
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(n + 1) +
           static_cast<std::size_t>(c);
  };

  for (int k = 0; k < n; ++k) {
    int piv = k;
    double best = std::fabs(a[id(k, k)]);
    for (int r = k + 1; r < n; ++r) {
      const double v = std::fabs(a[id(r, k)]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < eps) return false;

    if (piv != k) {
      for (int c = k; c <= n; ++c) std::swap(a[id(k, c)], a[id(piv, c)]);
    }

    const double diag = a[id(k, k)];
    for (int r = k + 1; r < n; ++r) {
      const double f = a[id(r, k)] / diag;
      if (std::fabs(f) < eps) continue;
      a[id(r, k)] = 0.0;
      for (int c = k + 1; c <= n; ++c) a[id(r, c)] -= f * a[id(k, c)];
    }
  }

  x.assign(static_cast<std::size_t>(n), 0.0);
  for (int k = n - 1; k >= 0; --k) {
    double rhs = a[id(k, n)];
    for (int c = k + 1; c < n; ++c) rhs -= a[id(k, c)] * x[static_cast<std::size_t>(c)];
    const double diag = a[id(k, k)];
    if (std::fabs(diag) < eps) return false;
    x[static_cast<std::size_t>(k)] = rhs / diag;
  }
  return true;
}

}  // namespace

class PeryashkinVGaussVStripFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType& p) {
    return Sanitize(std::to_string(std::get<0>(p)) + "_" + std::get<1>(p));
  }

 protected:
  void SetUp() override {
    const TestType params =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const std::string& file = std::get<1>(params);

    input_ = LoadSystem(file);
    if (!SolveDenseRef(input_.augmented_matrix, input_.n, expected_)) {
      throw std::runtime_error("Reference solver failed for: " + file);
    }
  }

  bool CheckTestOutputData(OutType& out) final {
    if (out.size() != expected_.size()) return false;
    const double tol = 1e-8;
    for (std::size_t i = 0; i < out.size(); ++i) {
      if (std::fabs(out[i] - expected_[i]) > tol) return false;
    }
    return true;
  }

  InType GetTestInputData() final { return input_; }

 private:
  InType input_;
  OutType expected_;
};

TEST_P(PeryashkinVGaussVStripFuncTests, SolveBandSystem) {
  ExecuteTest(GetParam());
}

namespace {

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(0, "test_0.txt"),
    std::make_tuple(1, "test_1.txt"),
    std::make_tuple(2, "test_2.txt"),
};

const auto kTasks = std::tuple_cat(
    ppc::util::AddFuncTask<PeryashkinVGaussVStripMPI, InType>(kTestParam, PPC_SETTINGS_peryashkin_v_gauss_vstrip),
    ppc::util::AddFuncTask<PeryashkinVGaussVStripSEQ, InType>(kTestParam, PPC_SETTINGS_peryashkin_v_gauss_vstrip));

const auto kValues = ppc::util::ExpandToValues(kTasks);
const auto kName = PeryashkinVGaussVStripFuncTests::PrintFuncTestName<PeryashkinVGaussVStripFuncTests>;

INSTANTIATE_TEST_SUITE_P(PeryashkinVGaussVStrip, PeryashkinVGaussVStripFuncTests, kValues, kName);

}  // namespace

}  // namespace peryashkin_v_gauss_vstrip
