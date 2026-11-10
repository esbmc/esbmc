#pragma once

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

// Singular values in descending order with unit left and right singular
// vectors u[k] and v[k] of a square matrix.
struct singular_system
{
  std::vector<double> sigma;
  matrix u, v;
};

// Throws a "TypeError: numpy.linalg.<function> ..." diagnostic when the
// spectrum is complex, or too close to repeated to bound the eigenvalues
// (and, for eig, the eigenvectors).
eigen_system real_eigen_system(const std::string &function, const matrix &m);

// The singular values only, which any rounding keeps well conditioned.
std::vector<double> singular_values_of(const matrix &m);

// Throws when the factors cannot be checked against m.
singular_system singular_system_of(const matrix &m);
} // namespace dense_spectrum
