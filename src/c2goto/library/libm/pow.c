
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

  /* here we know y is finite */

  int ye;
  double ym = frexp(y, &ye);
  int is_int = nearbyint(y) == y;
  int odd_int = ye > 0 && ye <= 53 && is_int && ((long long)y & 1);

  if (x == 0.0) {
    double r = signbit(y) ? HUGE_VAL : 0.0;
    return odd_int ? copysign(r, x) : r;
  }

  if (isinf(x)) {
    double r = signbit(y) ? 0.0 : INFINITY;
    return odd_int ? copysign(r, x) : r;
  }

  /* here we know x is finite */

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
