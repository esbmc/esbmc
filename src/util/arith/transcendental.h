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
/// an infinity.
struct domain
{
  double low, high;
  bool periodic;
};

bool is_bounded(const std::string &function);
bool is_odd(const std::string &function);
double host_value(const std::string &function, double x);
/// The results at @p x, a single value where every implementation is exact.
interval at(const std::string &function, double x);
const std::vector<piece> &pieces(const std::string &function);
domain domain_of(const std::string &function);
} // namespace transcendental
