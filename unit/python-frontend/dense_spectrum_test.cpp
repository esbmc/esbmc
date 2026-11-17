#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/numpy/dense_spectrum.h>

#include <cmath>
#include <complex>

using dense_spectrum::complex_vector;
using dense_spectrum::matrix;

namespace
{
// max_i |(A v - lambda v)_i|
double
residual(const matrix &a, std::complex<double> lambda, const complex_vector &v)
{
  double worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    std::complex<double> r = -lambda * v[i];
    for (std::size_t j = 0; j < a.size(); ++j)
      r += a[i][j] * v[j];
    worst = std::max(worst, std::abs(r));
  }
  return worst;
}

double norm_squared(const complex_vector &v)
{
  double sum = 0;
  for (const auto &x : v)
    sum += std::norm(x);
  return sum;
}
} // namespace

TEST_CASE("has_real_spectrum tells a complex pair apart", "[dense_spectrum]")
{
  REQUIRE(dense_spectrum::has_real_spectrum({{2, 1, 0}, {1, 3, 1}, {0, 1, 4}}));
  REQUIRE_FALSE(
    dense_spectrum::has_real_spectrum({{1, 2, 1}, {-2, 1, 0}, {1, 0, 3}}));
  REQUIRE_FALSE(dense_spectrum::has_real_spectrum({{1, -2}, {1, 3}}));
}

TEST_CASE(
  "complex_eigen_system pairs conjugates and checks its vectors",
  "[dense_spectrum]")
{
  const matrix a = {{1, 2, 1}, {-2, 1, 0}, {1, 0, 3}};
  const auto system = dense_spectrum::complex_eigen_system_of("eig", a);
  REQUIRE(system.values.size() == 3);
  std::size_t pairs = 0;
  for (std::size_t k = 0; k < 3; ++k)
  {
    const auto value = system.values[k];
    if (value.imag() > 0)
    {
      ++pairs;
      REQUIRE(system.values[k + 1] == std::conj(value));
    }
    REQUIRE(system.value_radius[k] > 0);
    REQUIRE(system.value_radius[k] < 1e-9);
    REQUIRE_FALSE(system.candidates[k].empty());
    for (const auto &candidate : system.candidates[k])
    {
      REQUIRE(residual(a, value, candidate.vector) < 1e-12);
      REQUIRE(std::fabs(norm_squared(candidate.vector) - 1) < 1e-12);
      if (value.imag() == 0)
        REQUIRE(candidate.real_component == 3);
      else
      {
        // LAPACK makes the largest component real.
        const auto &largest = candidate.vector[candidate.real_component];
        REQUIRE(largest.imag() == 0);
        for (const auto &x : candidate.vector)
          REQUIRE(std::abs(x) <= std::abs(largest) + 1e-9);
      }
    }
  }
  REQUIRE(pairs == 1);
  double trace = 0;
  for (const auto &value : system.values)
    trace += value.real();
  REQUIRE(std::fabs(trace - 5) < 1e-12);
}

TEST_CASE(
  "complex_eigen_system offers each tied largest component",
  "[dense_spectrum]")
{
  // The eigenvectors of a rotation have components of equal magnitude.
  const auto system =
    dense_spectrum::complex_eigen_system_of("eig", {{0, -1}, {1, 0}});
  REQUIRE(std::abs(system.values[0] - std::complex<double>(0, 1)) < 1e-15);
  REQUIRE(system.candidates[0].size() == 2);
  REQUIRE(system.candidates[1].size() == 2);
}

TEST_CASE("singular_system_of factors rectangular matrices", "[dense_spectrum]")
{
  const matrix wide = {{3, 1, 2}, {1, 4, 0}};
  SECTION("full factors complete the null space")
  {
    const auto system = dense_spectrum::singular_system_of(wide, true);
    REQUIRE(system.sigma.size() == 2);
    REQUIRE(system.u.size() == 2);
    REQUIRE(system.v.size() == 3);
    REQUIRE(system.sigma[0] > system.sigma[1]);
    for (std::size_t a = 0; a < 3; ++a)
      for (std::size_t b = 0; b < 3; ++b)
      {
        double dot = 0;
        for (std::size_t i = 0; i < 3; ++i)
          dot += system.v[a][i] * system.v[b][i];
        REQUIRE(std::fabs(dot - (a == b ? 1 : 0)) < 1e-12);
      }
  }
  SECTION("reduced factors of the transpose swap the sides")
  {
    const auto system =
      dense_spectrum::singular_system_of({{3, 1}, {1, 4}, {2, 0}}, false);
    REQUIRE(system.u.size() == 2);
    REQUIRE(system.u[0].size() == 3);
    REQUIRE(system.v.size() == 2);
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 2; ++j)
      {
        double sum = 0;
        for (std::size_t k = 0; k < 2; ++k)
          sum += system.u[k][i] * system.sigma[k] * system.v[k][j];
        const matrix tall = {{3, 1}, {1, 4}, {2, 0}};
        REQUIRE(std::fabs(sum - tall[i][j]) < 1e-12);
      }
  }
  SECTION("singular values match the factors' and the transpose's")
  {
    const auto sigma = dense_spectrum::singular_values_of(wide);
    const auto tall =
      dense_spectrum::singular_values_of({{3, 1}, {1, 4}, {2, 0}});
    REQUIRE(sigma.size() == 2);
    REQUIRE(std::fabs(sigma[0] - tall[0]) < 1e-12);
    REQUIRE(std::fabs(sigma[0] * sigma[0] + sigma[1] * sigma[1] - 31) < 1e-12);
  }
}

namespace
{
double dot(const std::vector<double> &a, const std::vector<double> &b)
{
  double sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    sum += a[i] * b[i];
  return sum;
}

// The rows of `vectors` are orthonormal and orthogonal to those of `known`.
void require_orthonormal(const matrix &vectors, const matrix &known = {})
{
  for (std::size_t a = 0; a < vectors.size(); ++a)
  {
    for (std::size_t b = 0; b < vectors.size(); ++b)
      REQUIRE(std::fabs(dot(vectors[a], vectors[b]) - (a == b)) < 1e-12);
    for (const auto &k : known)
      REQUIRE(std::fabs(dot(vectors[a], k)) < 1e-12);
  }
}

void require_reconstructs(
  const matrix &a,
  const dense_spectrum::singular_system &system)
{
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < a[0].size(); ++j)
    {
      double sum = 0;
      for (std::size_t k = 0; k < system.sigma.size(); ++k)
        sum += system.u[k][i] * system.sigma[k] * system.v[k][j];
      REQUIRE(std::fabs(sum - a[i][j]) < 1e-12);
    }
}
} // namespace

TEST_CASE(
  "singular_system_of completes rank deficient factors",
  "[dense_spectrum]")
{
  SECTION("a zero singular value of the reduced factors")
  {
    const matrix a = {{1, 2, 3}, {2, 4, 6}};
    const auto system = dense_spectrum::singular_system_of(a, false);
    REQUIRE(system.rank == 1);
    REQUIRE(system.u.size() == 2);
    REQUIRE(system.v.size() == 2);
    REQUIRE(system.sigma[1] < 1e-12);
    // The reduced Vh keeps one vector of a plane, which v_rest spans.
    REQUIRE(system.u_rest.size() == 1);
    REQUIRE(system.v_rest.size() == 2);
    require_orthonormal(system.u);
    require_orthonormal(system.v);
    require_orthonormal(system.v_rest, {system.v[0]});
    require_reconstructs(a, system);
  }
  SECTION("a full factor completing a plane, through the transpose")
  {
    const matrix a = {{1, 2, 3}};
    const auto system = dense_spectrum::singular_system_of(a, true);
    REQUIRE(system.rank == 1);
    REQUIRE(system.v.size() == 3);
    REQUIRE(system.v_rest.size() == 2);
    REQUIRE(system.u_rest.empty());
    require_orthonormal(system.v);
    require_orthonormal(system.v_rest, {system.v[0]});
    require_reconstructs(a, system);
  }
  SECTION("a full factor completing a plane directly")
  {
    const matrix a = {{1}, {2}, {2}};
    const auto system = dense_spectrum::singular_system_of(a, true);
    REQUIRE(system.u.size() == 3);
    REQUIRE(system.u_rest.size() == 2);
    require_orthonormal(system.u);
    require_orthonormal(system.u_rest, {system.u[0]});
    require_reconstructs(a, system);
  }
  SECTION("a zero matrix has rank zero and no gap to divide by")
  {
    const matrix a = {{0, 0}, {0, 0}, {0, 0}};
    const auto system = dense_spectrum::singular_system_of(a, true);
    REQUIRE(system.rank == 0);
    REQUIRE(system.sigma[0] == 0);
    REQUIRE(system.u_rest.size() == 3);
    require_orthonormal(system.u);
    require_orthonormal(system.v);
  }
}

TEST_CASE(
  "complex_input_eigen_system bounds a complex matrix's spectrum",
  "[dense_spectrum]")
{
  using cd = std::complex<double>;
  const auto residual_of = [](
                             const dense_spectrum::complex_matrix &a,
                             cd lambda,
                             const complex_vector &v) {
    double worst = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      cd r = -lambda * v[i];
      for (std::size_t j = 0; j < a.size(); ++j)
        r += a[i][j] * v[j];
      worst = std::max(worst, std::abs(r));
    }
    return worst;
  };
  SECTION("a 3x3 spectrum with no conjugate pairs")
  {
    const dense_spectrum::complex_matrix a = {
      {{1, 1}, {2, 0}, {0, 0}},
      {{0, 0}, {2, 0.5}, {1, 0}},
      {{0, 1}, {0, 0}, {3, 0}}};
    const auto system = dense_spectrum::complex_input_eigen_system("eig", a);
    REQUIRE(system.values.size() == 3);
    cd trace = 0;
    for (std::size_t k = 0; k < 3; ++k)
    {
      trace += system.values[k];
      REQUIRE(system.value_radius[k] > 0);
      REQUIRE(system.value_radius[k] < 1e-9);
      REQUIRE_FALSE(system.candidates[k].empty());
      for (const auto &candidate : system.candidates[k])
      {
        REQUIRE(residual_of(a, system.values[k], candidate.vector) < 1e-12);
        REQUIRE(std::fabs(norm_squared(candidate.vector) - 1) < 1e-12);
        const cd largest = candidate.vector[candidate.real_component];
        REQUIRE(largest.imag() == 0);
        for (const cd &x : candidate.vector)
          REQUIRE(std::abs(x) <= std::abs(largest) + 1e-9);
      }
    }
    REQUIRE(std::abs(trace - cd(6, 1.5)) < 1e-12);
  }
  SECTION("a real eigenvalue of a complex matrix has a complex vector")
  {
    const dense_spectrum::complex_matrix a = {
      {{1, 0}, {0, 0}}, {{5, 0}, {2, 1}}};
    const auto system = dense_spectrum::complex_input_eigen_system("eig", a);
    bool seen = false;
    for (std::size_t k = 0; k < 2; ++k)
      if (std::abs(system.values[k] - cd(1, 0)) < 1e-12)
      {
        seen = true;
        const auto &candidate = system.candidates[k].front();
        REQUIRE(candidate.real_component == 1);
        REQUIRE(candidate.vector[0].imag() != 0);
      }
    REQUIRE(seen);
  }
  SECTION("eigenvalues too close to bound are refused")
  {
    const dense_spectrum::complex_matrix a = {
      {{1, 0}, {3, 0}}, {{0, 0}, {1, 1e-9}}};
    REQUIRE_THROWS_AS(
      dense_spectrum::complex_input_eigen_system("eig", a), std::runtime_error);
  }
}

TEST_CASE(
  "symmetric_eigenvalues keeps a repeated eigenvalue's multiplicity",
  "[dense_spectrum]")
{
  const auto values =
    dense_spectrum::symmetric_eigenvalues({{2, 1, 1}, {1, 2, 1}, {1, 1, 2}});
  REQUIRE(values.size() == 3);
  REQUIRE(std::fabs(values[0] - 1) < 1e-15);
  REQUIRE(std::fabs(values[1] - 1) < 1e-15);
  REQUIRE(std::fabs(values[2] - 4) < 1e-15);
}

TEST_CASE("close spectra keep their computed values", "[dense_spectrum]")
{
  SECTION("symmetric eigenvalues a small gap apart stay apart")
  {
    const auto values = dense_spectrum::symmetric_eigenvalues(
      {{1, 1e-10, 0}, {1e-10, 1 + 1e-9, 0}, {0, 0, 3}});
    REQUIRE(values[1] - values[0] > 9e-10);
    REQUIRE(std::fabs(values[0] + values[1] - (2 + 1e-9)) < 1e-15);
  }
  SECTION("a repeated singular value's computed plane reconstructs A")
  {
    const matrix a = {{4, 4, 1}, {-4, 2, 2}, {2, -4, 2}};
    const auto system = dense_spectrum::singular_system_of(a, true);
    REQUIRE(std::fabs(system.sigma[0] - system.sigma[1]) < 1e-12);
    require_orthonormal(system.u);
    require_orthonormal(system.v);
    require_reconstructs(a, system);
  }
}
