#include <python-frontend/numpy/dense_spectrum.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>

namespace dense_spectrum
{
namespace
{
using real = long double;
using vec = std::vector<real>;
using mat = std::vector<vec>;

// LAPACK's eigen and singular value routines are backward stable: their
// results are exact for a matrix within this multiple of ||A|| of A.
constexpr real backward_error = 1e-12L;
// The relations the extended-precision results must satisfy.
constexpr real check_tolerance = 1e-15L;

mat widen(const matrix &m)
{
  mat out;
  for (const auto &row : m)
    out.emplace_back(row.begin(), row.end());
  return out;
}

real frobenius(const mat &m)
{
  real sum = 0;
  for (const vec &row : m)
    for (real x : row)
      sum += x * x;
  return std::sqrt(sum);
}

real dot(const vec &a, const vec &b)
{
  return std::inner_product(a.begin(), a.end(), b.begin(), real(0));
}

vec unit(vec v)
{
  const real length = std::sqrt(dot(v, v));
  for (real &x : v)
    x /= length;
  return v;
}

mat transpose(const mat &m)
{
  mat t(m[0].size(), vec(m.size()));
  for (std::size_t i = 0; i < m.size(); ++i)
    for (std::size_t j = 0; j < m[i].size(); ++j)
      t[j][i] = m[i][j];
  return t;
}

// x with (m - shift I) x = b, by elimination with partial pivoting; a zero
// pivot is replaced by a tiny one, as inverse iteration allows.
vec shifted_solve(mat m, real shift, vec b)
{
  const std::size_t n = m.size();
  for (std::size_t i = 0; i < n; ++i)
    m[i][i] -= shift;
  for (std::size_t k = 0; k < n; ++k)
  {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::fabs(m[i][k]) > std::fabs(m[p][k]))
        p = i;
    std::swap(m[p], m[k]);
    std::swap(b[p], b[k]);
    if (m[k][k] == 0)
      m[k][k] = 1e-30L;
    for (std::size_t i = k + 1; i < n; ++i)
    {
      const real f = m[i][k] / m[k][k];
      for (std::size_t j = k; j < n; ++j)
        m[i][j] -= f * m[k][j];
      b[i] -= f * b[k];
    }
  }
  vec x(n);
  for (std::size_t i = n; i-- > 0;)
  {
    real s = b[i];
    for (std::size_t j = i + 1; j < n; ++j)
      s -= m[i][j] * x[j];
    x[i] = s / m[i][i];
  }
  return x;
}

// The characteristic polynomial's coefficients, lowest degree first, by
// Faddeev-LeVerrier.
vec characteristic_polynomial(const mat &a)
{
  const std::size_t n = a.size();
  vec c(n + 1, 0);
  c[n] = 1;
  mat m(n, vec(n, 0));
  for (std::size_t k = 1; k <= n; ++k)
  {
    mat next(n, vec(n, 0));
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
      {
        for (std::size_t l = 0; l < n; ++l)
          next[i][j] += a[i][l] * m[l][j];
        if (i == j)
          next[i][j] += c[n - k + 1];
      }
    real trace = 0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t l = 0; l < n; ++l)
        trace += a[i][l] * next[l][i];
    c[n - k] = -trace / static_cast<real>(k);
    m = next;
  }
  return c;
}

std::complex<real> evaluate(const vec &c, std::complex<real> z)
{
  std::complex<real> value = 0;
  for (std::size_t i = c.size(); i-- > 0;)
    value = value * z + c[i];
  return value;
}

// The roots of a monic polynomial, by Durand-Kerner.
std::vector<std::complex<real>> roots(const vec &c)
{
  const std::size_t n = c.size() - 1;
  std::vector<std::complex<real>> z(n);
  const std::complex<real> seed(0.4L, 0.9L);
  for (std::size_t i = 0; i < n; ++i)
    z[i] = std::pow(seed, static_cast<int>(i));
  for (int iteration = 0; iteration < 1000; ++iteration)
    for (std::size_t i = 0; i < n; ++i)
    {
      std::complex<real> denominator = 1;
      for (std::size_t j = 0; j < n; ++j)
        if (j != i)
          denominator *= z[i] - z[j];
      z[i] -= evaluate(c, z[i]) / denominator;
    }
  return z;
}

// Newton's method on det(A - lambda I), whose logarithmic derivative is
// -trace((A - lambda I)^-1).
real refine(const mat &a, real lambda)
{
  const std::size_t n = a.size();
  for (int iteration = 0; iteration < 20; ++iteration)
  {
    real trace = 0;
    for (std::size_t j = 0; j < n; ++j)
    {
      vec e(n, 0);
      e[j] = 1;
      trace += shifted_solve(a, lambda, e)[j];
    }
    const real step = 1 / trace;
    lambda += step;
    if (!(std::fabs(step) > 1e-21L))
      break;
  }
  return lambda;
}

vec eigenvector(const mat &a, real lambda)
{
  vec v(a.size(), 1);
  for (int iteration = 0; iteration < 4; ++iteration)
    v = unit(shifted_solve(a, lambda, v));
  return v;
}

real residual(const mat &a, real lambda, const vec &v)
{
  real sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    const real r = dot(a[i], v) - lambda * v[i];
    sum += r * r;
  }
  return std::sqrt(sum);
}

[[noreturn]] void
reject_spectrum(const std::string &function, const std::string &reason)
{
  throw std::runtime_error(
    "TypeError: numpy.linalg." + function +
    " of a matrix larger than 2x2 with " + reason + " is not supported");
}

// The eigenvalues of `a` (scaled to unit norm), which must all be real.
vec real_eigenvalues(const std::string &function, const mat &a)
{
  vec values;
  for (const std::complex<real> &root : roots(characteristic_polynomial(a)))
  {
    if (std::fabs(root.imag()) > 1e-6L)
      reject_spectrum(function, "complex eigenvalues");
    values.push_back(refine(a, root.real()));
  }
  std::sort(values.begin(), values.end());
  return values;
}

real smallest_gap(const vec &values)
{
  real gap = INFINITY;
  for (std::size_t k = 0; k + 1 < values.size(); ++k)
    gap = std::min(gap, values[k + 1] - values[k]);
  return gap;
}
} // namespace

eigen_system real_eigen_system(const std::string &function, const matrix &m)
{
  const mat wide = widen(m);
  const real norm = frobenius(wide);
  mat a = wide;
  for (vec &row : a)
    for (real &x : row)
      x /= norm;
  const vec values = real_eigenvalues(function, a);
  const real gap = smallest_gap(values);
  const mat at = transpose(a);
  eigen_system system;
  real conditions = 0;
  std::vector<real> condition;
  for (real lambda : values)
  {
    const vec v = eigenvector(a, lambda), w = eigenvector(at, lambda);
    if (!(residual(a, lambda, v) <= check_tolerance))
      reject_spectrum(function, "an eigenvector that cannot be checked");
    condition.push_back(1 / std::fabs(dot(w, v)));
    conditions += condition.back();
    system.values.push_back(static_cast<double>(lambda * norm));
    system.vectors.emplace_back(v.begin(), v.end());
  }
  // First-order perturbation bounds, valid while the eigenvalues stay
  // separated: |d lambda_k| <= kappa_k ||E||, and each eigenvector moves by
  // at most sum_j kappa_j ||E|| / gap.
  if (!(4 * backward_error * conditions < gap))
    reject_spectrum(function, "repeated or ill-conditioned eigenvalues");
  for (real kappa : condition)
    system.value_radius.push_back(
      static_cast<double>(2 * backward_error * kappa * norm));
  const real vector_radius =
    2 * backward_error * conditions / gap + backward_error;
  if (function == "eig" && vector_radius > 1e-6L)
    reject_spectrum(function, "ill-conditioned eigenvectors");
  system.vector_radius = static_cast<double>(vector_radius);
  return system;
}

namespace
{
using cplx = std::complex<real>;
using cvec = std::vector<cplx>;

// x with (m - shift I) x = b for a complex shift, as shifted_solve.
cvec complex_shifted_solve(const mat &real_m, cplx shift, cvec b)
{
  const std::size_t n = real_m.size();
  std::vector<cvec> m(n, cvec(n));
  for (std::size_t i = 0; i < n; ++i)
  {
    for (std::size_t j = 0; j < n; ++j)
      m[i][j] = real_m[i][j];
    m[i][i] -= shift;
  }
  for (std::size_t k = 0; k < n; ++k)
  {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::abs(m[i][k]) > std::abs(m[p][k]))
        p = i;
    std::swap(m[p], m[k]);
    std::swap(b[p], b[k]);
    if (m[k][k] == cplx(0))
      m[k][k] = 1e-30L;
    for (std::size_t i = k + 1; i < n; ++i)
    {
      const cplx f = m[i][k] / m[k][k];
      for (std::size_t j = k; j < n; ++j)
        m[i][j] -= f * m[k][j];
      b[i] -= f * b[k];
    }
  }
  cvec x(n);
  for (std::size_t i = n; i-- > 0;)
  {
    cplx sum = b[i];
    for (std::size_t j = i + 1; j < n; ++j)
      sum -= m[i][j] * x[j];
    x[i] = sum / m[i][i];
  }
  return x;
}

cvec complex_unit(cvec v)
{
  real sum = 0;
  for (const cplx &x : v)
    sum += std::norm(x);
  const real length = std::sqrt(sum);
  for (cplx &x : v)
    x /= length;
  return v;
}

// Newton's method on det(A - lambda I), as refine, for a complex root.
cplx complex_refine(const mat &a, cplx lambda)
{
  const std::size_t n = a.size();
  for (int iteration = 0; iteration < 20; ++iteration)
  {
    cplx trace = 0;
    for (std::size_t j = 0; j < n; ++j)
    {
      cvec e(n, 0);
      e[j] = 1;
      trace += complex_shifted_solve(a, lambda, e)[j];
    }
    const cplx step = cplx(1) / trace;
    lambda += step;
    if (!(std::abs(step) > 1e-21L))
      break;
  }
  return lambda;
}

cvec complex_eigenvector(const mat &a, cplx lambda)
{
  cvec v(a.size(), 1);
  for (int iteration = 0; iteration < 4; ++iteration)
    v = complex_unit(complex_shifted_solve(a, lambda, v));
  return v;
}

real complex_residual(const mat &a, cplx lambda, const cvec &v)
{
  real sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
  {
    cplx r = -lambda * v[i];
    for (std::size_t j = 0; j < a.size(); ++j)
      r += a[i][j] * v[j];
    sum += std::norm(r);
  }
  return std::sqrt(sum);
}

// The eigenvalues of `a` (scaled to unit norm): the real ones refined as
// real numbers, and each complex pair as its upper member and conjugate.
cvec spectrum(const std::string &function, const mat &a)
{
  cvec values;
  for (const cplx &root : roots(characteristic_polynomial(a)))
  {
    if (std::fabs(root.imag()) <= 1e-6L)
      values.emplace_back(refine(a, root.real()), 0);
    else if (root.imag() > 0)
    {
      const cplx lambda = complex_refine(a, root);
      values.push_back(lambda);
      values.push_back(std::conj(lambda));
    }
  }
  if (values.size() != a.size())
    reject_spectrum(function, "eigenvalues that cannot be paired");
  return values;
}

// LAPACK scales each eigenvector of a complex eigenvalue so that its
// largest component is real: one normalisation of `v` per component whose
// magnitude is within `slack` of the largest, that component made real.
std::vector<eigenvector_candidate> phase_candidates(const cvec &v, real slack)
{
  real largest = 0;
  for (const cplx &x : v)
    largest = std::max(largest, std::norm(x));
  std::vector<eigenvector_candidate> candidates;
  for (std::size_t k = 0; k < v.size(); ++k)
  {
    if (std::norm(v[k]) < largest - slack)
      continue;
    const cplx phase = std::conj(v[k]) / std::abs(v[k]);
    complex_vector candidate;
    for (const cplx &x : v)
      candidate.emplace_back(x * phase);
    candidate[k] = {static_cast<double>(std::abs(v[k])), 0.0};
    candidates.push_back({candidate, k});
  }
  return candidates;
}

std::vector<eigenvector_candidate>
conjugates(const std::vector<eigenvector_candidate> &candidates)
{
  std::vector<eigenvector_candidate> out = candidates;
  for (eigenvector_candidate &candidate : out)
    for (std::complex<double> &x : candidate.vector)
      x = std::conj(x);
  return out;
}

// The eigenvector candidates of the k-th eigenvalue, given the unit vector
// inverse iteration found for it.
std::vector<eigenvector_candidate> eigenvector_candidates(
  const cvec &values,
  std::size_t k,
  const cvec &v,
  const std::vector<std::vector<eigenvector_candidate>> &earlier,
  real slack)
{
  if (values[k].imag() < 0)
    return conjugates(earlier.back());
  if (values[k].imag() > 0)
    return phase_candidates(v, slack);
  complex_vector real_vector;
  real length = 0;
  for (const cplx &x : v)
    length += x.real() * x.real();
  for (const cplx &x : v)
    real_vector.emplace_back(
      static_cast<double>(x.real() / std::sqrt(length)), 0.0);
  return {{real_vector, v.size()}};
}
} // namespace

bool has_real_spectrum(const matrix &m)
{
  const mat wide = widen(m);
  const real norm = frobenius(wide);
  mat a = wide;
  for (vec &row : a)
    for (real &x : row)
      x /= norm;
  for (const std::complex<real> &root : roots(characteristic_polynomial(a)))
    if (std::fabs(root.imag()) > 1e-6L)
      return false;
  return true;
}

complex_eigen_system
complex_eigen_system_of(const std::string &function, const matrix &m)
{
  const mat wide = widen(m);
  const real norm = frobenius(wide);
  mat a = wide;
  for (vec &row : a)
    for (real &x : row)
      x /= norm;
  const cvec values = spectrum(function, a);
  real gap = INFINITY;
  for (std::size_t i = 0; i < values.size(); ++i)
    for (std::size_t j = i + 1; j < values.size(); ++j)
      gap = std::min(gap, std::abs(values[i] - values[j]));
  const mat at = transpose(a);
  std::vector<cvec> vectors;
  std::vector<real> condition;
  real conditions = 0;
  for (const cplx &lambda : values)
  {
    const cvec v = complex_eigenvector(a, lambda),
               w = complex_eigenvector(at, lambda);
    if (!(complex_residual(a, lambda, v) <= check_tolerance))
      reject_spectrum(function, "an eigenvector that cannot be checked");
    cplx product = 0;
    for (std::size_t i = 0; i < v.size(); ++i)
      product += w[i] * v[i];
    condition.push_back(1 / std::abs(product));
    conditions += condition.back();
    vectors.push_back(v);
  }
  if (!(4 * backward_error * conditions < gap))
    reject_spectrum(function, "repeated or ill-conditioned eigenvalues");
  // Making the largest component real turns an error r in each component
  // into at most r (1 + sqrt(n)) more, as that component is at least
  // 1 / sqrt(n).
  const real vector_radius =
    (2 * backward_error * conditions / gap + backward_error) *
    (2 + 2 * std::sqrt(static_cast<real>(m.size())));
  if (function == "eig" && vector_radius > 1e-6L)
    reject_spectrum(function, "ill-conditioned eigenvectors");
  complex_eigen_system system;
  system.vector_radius = static_cast<double>(vector_radius);
  for (std::size_t k = 0; k < values.size(); ++k)
  {
    const cplx value = values[k] * norm;
    system.values.emplace_back(
      static_cast<double>(value.real()), static_cast<double>(value.imag()));
    system.value_radius.push_back(
      static_cast<double>(2 * backward_error * condition[k] * norm));
    system.candidates.push_back(eigenvector_candidates(
      values, k, vectors[k], system.candidates, 8 * vector_radius));
  }
  return system;
}

namespace
{
// One-sided Jacobi: rotates the columns of w (initially A) until they are
// orthogonal, accumulating the rotations in v, so that A v = w.
void orthogonalise_columns(mat &w, mat &v)
{
  const std::size_t n = w[0].size();
  for (int sweep = 0; sweep < 60; ++sweep)
  {
    bool rotated = false;
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = p + 1; q < n; ++q)
      {
        real alpha = 0, beta = 0, gamma = 0;
        for (std::size_t i = 0; i < w.size(); ++i)
        {
          alpha += w[i][p] * w[i][p];
          beta += w[i][q] * w[i][q];
          gamma += w[i][p] * w[i][q];
        }
        if (std::fabs(gamma) <= 1e-21L * std::sqrt(alpha * beta))
          continue;
        rotated = true;
        const real zeta = (beta - alpha) / (2 * gamma);
        const real t =
          (zeta >= 0 ? 1 : -1) / (std::fabs(zeta) + std::sqrt(1 + zeta * zeta));
        const real c = 1 / std::sqrt(1 + t * t), s = c * t;
        for (mat *x : {&w, &v})
          for (vec &row : *x)
          {
            const real xp = row[p], xq = row[q];
            row[p] = c * xp - s * xq;
            row[q] = s * xp + c * xq;
          }
      }
    if (!rotated)
      return;
  }
}

struct jacobi_result
{
  vec sigma;
  mat u, v; // columns as rows: u[k], v[k]
};

// A matrix with at least as many rows as columns.
jacobi_result jacobi_svd(const matrix &m)
{
  const std::size_t n = m[0].size();
  mat w = widen(m), v(n, vec(n, 0));
  for (std::size_t i = 0; i < n; ++i)
    v[i][i] = 1;
  orthogonalise_columns(w, v);
  const mat wt = transpose(w), vt = transpose(v);
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::vector<real> norms;
  for (const vec &column : wt)
    norms.push_back(std::sqrt(dot(column, column)));
  std::stable_sort(order.begin(), order.end(), [&](auto i, auto j) {
    return norms[i] > norms[j];
  });
  jacobi_result result;
  for (std::size_t k : order)
  {
    result.sigma.push_back(norms[k]);
    result.u.push_back(wt[k]);
    result.v.push_back(vt[k]);
  }
  return result;
}

// The unit vector orthogonal to the first n - 1 rows of `basis`, on the side
// `hint` points to.
vec completion(const mat &basis, const vec &hint)
{
  const std::size_t n = hint.size();
  vec best;
  real best_length = -1;
  for (std::size_t e = 0; e < n; ++e)
  {
    vec x(n, 0);
    x[e] = 1;
    for (int pass = 0; pass < 2; ++pass)
      for (std::size_t k = 0; k + 1 < n; ++k)
      {
        const real projection = dot(x, basis[k]);
        for (std::size_t i = 0; i < n; ++i)
          x[i] -= projection * basis[k][i];
      }
    const real length = std::sqrt(dot(x, x));
    if (length > best_length)
    {
      best = x;
      best_length = length;
    }
  }
  best = unit(best);
  if (dot(best, hint) < 0)
    for (real &x : best)
      x = -x;
  return best;
}

real reconstruction_error(const mat &a, const jacobi_result &r)
{
  real worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < a[i].size(); ++j)
    {
      real sum = 0;
      for (std::size_t k = 0; k < r.sigma.size(); ++k)
        sum += r.u[k][i] * r.sigma[k] * r.v[k][j];
      worst = std::max(worst, std::fabs(sum - a[i][j]));
    }
  return worst;
}

real orthogonality_error(const mat &rows)
{
  real worst = 0;
  for (std::size_t i = 0; i < rows.size(); ++i)
    for (std::size_t j = 0; j < rows.size(); ++j)
      worst =
        std::max(worst, std::fabs(dot(rows[i], rows[j]) - (i == j ? 1 : 0)));
  return worst;
}

matrix narrow(const mat &m)
{
  matrix out;
  for (const vec &row : m)
    out.emplace_back(row.begin(), row.end());
  return out;
}

matrix transposed(const matrix &m)
{
  return narrow(transpose(widen(m)));
}

[[noreturn]] void reject_factors(const std::string &reason)
{
  throw std::runtime_error(
    "TypeError: numpy.linalg.svd factors of " + reason + " are not supported");
}

// The left singular vectors of a matrix with at least as many rows as
// columns: A v / sigma, which loses accuracy as sigma shrinks, so the last
// one of a square matrix is the unit vector the others fix up to its sign.
// A full factor adds the unit vector orthogonal to them, which is unique up
// to its sign only when it completes a basis on its own.
void left_vectors(jacobi_result &r, std::size_t rows, bool full)
{
  const std::size_t n = r.sigma.size();
  for (std::size_t k = 0; k + 1 < n; ++k)
    r.u[k] = unit(r.u[k]);
  if (rows == n)
  {
    r.u[n - 1] = completion(r.u, r.u[n - 1]);
    return;
  }
  if (!(r.sigma[n - 1] > 1e-12L * r.sigma[0]))
    reject_factors("a non-square matrix with a zero singular value");
  r.u[n - 1] = unit(r.u[n - 1]);
  if (!full)
    return;
  if (rows - n > 1)
    reject_factors(
      "full_matrices=True with more than one row or column past the square");
  r.u.push_back(completion(r.u, vec(rows, 0)));
}
} // namespace

std::vector<double> singular_values_of(const matrix &m)
{
  const vec sigma =
    jacobi_svd(m.size() < m[0].size() ? transposed(m) : m).sigma;
  return std::vector<double>(sigma.begin(), sigma.end());
}

singular_system singular_system_of(const matrix &m, bool full)
{
  if (m.size() < m[0].size())
  {
    singular_system system = singular_system_of(transposed(m), full);
    std::swap(system.u, system.v);
    return system;
  }
  jacobi_result r = jacobi_svd(m);
  left_vectors(r, m.size(), full);
  const mat a = widen(m);
  const real norm = frobenius(a);
  if (
    !(reconstruction_error(a, r) <= check_tolerance * norm) ||
    !(orthogonality_error(r.u) <= check_tolerance) ||
    !(orthogonality_error(r.v) <= check_tolerance))
    throw std::runtime_error(
      "TypeError: numpy.linalg.svd factors of this matrix cannot be checked");
  return {
    std::vector<double>(r.sigma.begin(), r.sigma.end()),
    narrow(r.u),
    narrow(r.v)};
}
} // namespace dense_spectrum
