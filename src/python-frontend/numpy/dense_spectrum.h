#pragma once

#include <complex>
#include <string>
#include <vector>

// Spectra of small dense real matrices, computed in extended precision and
// checked against their defining relations, with the radius within which
// LAPACK's backward-stable result may differ from them.
namespace dense_spectrum
{
using matrix = std::vector<std::vector<double>>;

// Real, distinct eigenvalues with unit right eigenvectors (vectors[k] belongs
// to values[k]). value_radius[k] bounds LAPACK's error on values[k] and
// vector_radius its error on each eigenvector component.
struct eigen_system
{
  std::vector<double> values;
  std::vector<double> value_radius;
  matrix vectors;
  double vector_radius;
};

using complex_vector = std::vector<std::complex<double>>;

// The eigenvalues of a real matrix with a complex conjugate pair, each pair
// adjacent with its positive imaginary part first (real eigenvalues have a
// zero one). candidates[k] lists the unit eigenvectors LAPACK may return for
// values[k]: a real one for a real eigenvalue; for a complex one, one per
// component that may be the largest, which LAPACK makes real (the second of
// a pair has the conjugates of the first's).
// An eigenvector LAPACK may return, with the component whose imaginary part
// it sets to exactly zero (vector.size() for a real vector).
struct eigenvector_candidate
{
  complex_vector vector;
  std::size_t real_component;
};

struct complex_eigen_system
{
  std::vector<std::complex<double>> values;
  std::vector<double> value_radius;
  std::vector<std::vector<eigenvector_candidate>> candidates;
  double vector_radius;
};

// Singular values in descending order with unit left and right singular
// vectors u[k] and v[k]: min(m, n) of each for the reduced factors, and m
// and n for the full ones. The first `rank` singular values are above
// rounding; the vectors past them only complete an orthonormal basis, and
// LAPACK may return any other basis of the space they span, of which
// u_rest and v_rest are one (m - rank and n - rank vectors, even for the
// reduced factors).
struct singular_system
{
  std::vector<double> sigma;
  matrix u, v;
  std::size_t rank;
  matrix u_rest, v_rest;
};

// Throws a "TypeError: numpy.linalg.<function> ..." diagnostic when the
// spectrum is complex, or too close to repeated to bound the eigenvalues
// (and, for eig, the eigenvectors).
eigen_system real_eigen_system(const std::string &function, const matrix &m);

// The eigenvalues of a symmetric matrix, ascending, each repeated as often
// as its multiplicity.
std::vector<double> symmetric_eigenvalues(const matrix &m);

// Whether every eigenvalue of `m` is real (up to the separation the dense
// routes need to tell a complex pair from a repeated real eigenvalue).
bool has_real_spectrum(const matrix &m);

// Throws, as real_eigen_system does, when the spectrum is too close to
// repeated to bound it.
complex_eigen_system
complex_eigen_system_of(const std::string &function, const matrix &m);

using complex_matrix = std::vector<std::vector<std::complex<double>>>;

// The eigenvalues of a complex matrix, in no particular order and with no
// conjugate pairs, each with candidates[k] as complex_eigen_system_of gives
// them for a complex eigenvalue. Throws as complex_eigen_system_of does.
complex_eigen_system complex_input_eigen_system(
  const std::string &function,
  const complex_matrix &m);

// The singular values only, which any rounding keeps well conditioned.
std::vector<double> singular_values_of(const matrix &m);

// Throws when the factors cannot be checked against m.
singular_system singular_system_of(const matrix &m, bool full = true);
} // namespace dense_spectrum
