#pragma once

#include <string>
#include <vector>

/// Bounds on the transcendental functions numpy.* and math.* share (NumPy
/// spelling): any double within --python-math-ulps (8) of the C library's value
/// and inside the function's range, exact where every implementation is
/// (ADR-NP-005).
namespace transcendental
{
struct interval
{
  double low, high;
};

/// The results for an argument in [from, to].
struct piece
{
  double from, to;
  interval result;
};

/// Outside [low, high] the result is NaN, and a periodic function is NaN at
/// an infinity. At a pole the result is an infinity, where math.* raises.
struct domain
{
  double low, high;
  bool periodic;
  std::vector<double> poles;
};

bool is_bounded(const std::string &function);
/// math.gamma and math.lgamma, which CPython computes itself rather than
/// the C library: a literal argument is bounded rather than exact, and NumPy
/// has no such function.
bool cpython_implements(const std::string &function);
bool is_odd(const std::string &function);
/// f(+0) = +0 and f(-0) = -0.
bool keeps_signed_zero(const std::string &function);
double host_value(const std::string &function, double x);
/// The results at @p x, a single value where every implementation is exact.
interval at(const std::string &function, double x);
const std::vector<piece> &pieces(const std::string &function);
domain domain_of(const std::string &function);

/// --python-math-ulps.
double max_ulps();
/// hypot, arctan2 and pow, bounded by relations between their arguments
/// rather than by pieces.
bool is_bounded_binary(const std::string &function);
double host_value(const std::string &function, double a, double b);
/// The results within --python-math-ulps of @p value, on its side of zero.
interval within_ulps(double value);
} // namespace transcendental
