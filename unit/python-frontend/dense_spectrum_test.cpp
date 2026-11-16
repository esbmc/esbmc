#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <python-frontend/numpy/dense_spectrum.h>

#include <cmath>
#include <complex>
#include <stdexcept>

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

TEST_CASE(
  "singular_system_of refuses factors LAPACK may rotate",
  "[dense_spectrum]")
{
  // Rank 1: the second right singular vector of the reduced factors lies
  // anywhere in a plane.
  REQUIRE_THROWS_AS(
    dense_spectrum::singular_system_of({{1, 2, 3}, {2, 4, 6}}, false),
    std::runtime_error);
  // A 1x3 matrix's full Vh completes a plane.
  REQUIRE_THROWS_AS(
    dense_spectrum::singular_system_of({{1, 2, 3}}, true), std::runtime_error);
  REQUIRE_NOTHROW(dense_spectrum::singular_system_of({{1, 2, 3}}, false));
}
