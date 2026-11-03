#include <math.h>

double atanh(double x)
{
__ESBMC_HIDE:;
  // C11 F.10.2.3: atanh(±0) is ±0, atanh(±1) is ±inf, and |x| > 1 is NaN.
  // Tiny arguments return x, as log of a value next to 1 would round to 0.
  double a = fabs(x);
  if (isnan(x) || a < 0x1p-28)
    return x;
  if (a > 1.0)
    return NAN;
  if (a == 1.0)
    return signbit(x) ? -INFINITY : INFINITY;
  // atanh is odd (C11 F.10p3); the two ranges follow glibc's e_atanh.c and
  // use log1p to keep the precision log loses near 1.
  double r = a < 0.5 ? 0.5 * log1p(2.0 * a + 2.0 * a * a / (1.0 - a))
                     : 0.5 * log1p(2.0 * a / (1.0 - a));
  return signbit(x) ? -r : r;
}
