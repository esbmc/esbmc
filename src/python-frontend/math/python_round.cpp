#include <python-frontend/math/python_round.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// value - floor(value) is exact for a double, so a tie is exactly 0.5.
double round_ties_to_even(const double value)
{
  const double lower = std::floor(value);
  const double diff = value - lower;
  if (diff < 0.5)
    return lower;
  if (diff > 0.5)
    return lower + 1.0;
  return std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0;
}

// CPython rounds the exact binary value half to even at the ndigits-th decimal
// digit and reads the decimal result back (float_round_impl in
// Objects/floatobject.c). printf's %f conversion of a double is exact.
// Answers nullopt where CPython raises OverflowError.
std::optional<double>
round_to_ndigits_ties_even(const double value, const int ndigits)
{
  if (!std::isfinite(value) || ndigits > 323)
    return value;
  if (ndigits < -308)
    return 0.0 * value;

  if (ndigits >= 0)
  {
    std::vector<char> buf(
      std::snprintf(nullptr, 0, "%.*f", ndigits, value) + 1);
    std::snprintf(buf.data(), buf.size(), "%.*f", ndigits, value);
    return std::strtod(buf.data(), nullptr);
  }

  // Round the integer digits at 10**-ndigits; a nonzero fraction breaks a tie
  // away from zero.
  const double magnitude = std::fabs(value);
  const double whole = std::trunc(magnitude);
  std::vector<char> buf(std::snprintf(nullptr, 0, "%.0f", whole) + 1);
  std::snprintf(buf.data(), buf.size(), "%.0f", whole);
  const std::size_t k = -static_cast<long>(ndigits);
  std::string digits(buf.data());
  if (digits.size() <= k)
    digits.insert(0, k + 1 - digits.size(), '0');

  const std::size_t keep = digits.size() - k;
  const std::string half = "5" + std::string(k - 1, '0');
  const int cmp = digits.compare(keep, k, half);
  const bool round_up =
    cmp > 0 ||
    (cmp == 0 && (magnitude != whole || (digits[keep - 1] - '0') % 2 != 0));

  digits.replace(keep, k, k, '0');
  if (round_up)
  {
    std::size_t i = keep;
    while (i > 0 && digits[i - 1] == '9')
      digits[--i] = '0';
    if (i == 0)
      digits.insert(0, 1, '1');
    else
      ++digits[i - 1];
  }
  const double rounded = std::strtod(digits.c_str(), nullptr);
  if (std::isinf(rounded))
    return std::nullopt;
  return std::signbit(value) ? -rounded : rounded;
}
