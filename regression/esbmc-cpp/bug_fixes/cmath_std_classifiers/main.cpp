/* Regression for https://github.com/Yiannis128/esbmc/issues/5
 *
 * std::isinf failed to parse: the C <math.h> classifier macros (e.g. glibc
 * lowers isinf to __builtin_isinf_sign) leaked into the std:: call, so
 * std::isinf(d) macro-expanded to std::__builtin_isinf_sign(d), which is not a
 * member of std. std::isnan already worked; isinf, isfinite and signbit did
 * not. The fix undefines the macros and provides std:: overloads for all five
 * classifiers, including the standard integral-argument overload (treated as
 * double).
 *
 * Expected: VERIFICATION SUCCESSFUL
 */
#include <cmath>
#include <cassert>

bool is_inf(double d)
{
  return std::isinf(d);
}

int main()
{
  double inf = INFINITY;
  double fin = 3.5;

  /* double overloads */
  assert(is_inf(inf));
  assert(!is_inf(fin));
  assert(!std::isnan(fin));
  assert(std::isfinite(fin));
  assert(!std::isfinite(inf));
  assert(std::isnormal(fin));
  assert(std::signbit(-fin));
  assert(!std::signbit(fin));

  /* float and long double overloads */
  assert(std::isinf((float)inf));
  assert(std::isfinite((long double)fin));

  /* integral argument is treated as double */
  int n = 5, m = -5;
  assert(!std::isinf(n));
  assert(std::isfinite(n));
  assert(std::signbit(m));
  assert(!std::signbit(n));

  return 0;
}
