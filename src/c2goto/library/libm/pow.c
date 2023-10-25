
#include <math.h>

double pow(double x, double y)
{
__ESBMC_HIDE:;
  if (x == 1.0 || y == 0.0)
    return 1.0;

  if (isnan(x))
    return x;
  if (isnan(y))
    return y;

  if (isinf(y)) {
    if (x == -1.0)
      return 1.0;
    x = fabs(x);
    if ((x < 1) == signbit(y)) /* x < 1 && y < 0 || x > 1 && y > 0 */
      return INFINITY;
    return +0.0;
  }

  __ESBMC_assert(isfinite(y), "");
  int ye;
  double ym = frexp(y, &ye);
  int is_int = nearbyint(y) == y;
  int odd_int = ye > 0 && ye <= 53 && is_int && ((long long)y & 1);

  if (x == 0.0) {
    if (signbit(y))
      return odd_int ? copysign(HUGE_VAL, x) : HUGE_VAL;
    if (odd_int)
      return copysign(0.0, x);
    return 0.0;
  }

  if (isinf(x)) {
    if (signbit(x) && odd_int)
      return signbit(y) ? -0.0 : -INFINITY;
    return signbit(y) ? +0.0 : INFINITY;
  }

  __ESBMC_assert(isfinite(x), "");

  if (signbit(x) && !is_int)
    return NAN;

  /* TODO: for integer exponents (when is_int) we could use an iterative
   * approach, e.g., exponentiating by squaring for increased accuracy */

  double l = log(fabs(x));
  double r = exp(l * y);
  return odd_int ? copysign(r, x) : r;
}

double __pow(double base, double exponent)
{
__ESBMC_HIDE:;
  return pow(base, exponent);
}
