#include <math.h>

static double scaled_hypot(double x, double y, double scale, double unscale)
{
  x *= scale;
  y *= scale;
  return sqrt(x * x + y * y) * unscale;
}

double hypot(double x, double y)
{
__ESBMC_HIDE:;
  // C11 F.10.4.3: an infinite argument gives +inf even when the other is NaN.
  if (isinf(x) || isinf(y))
    return INFINITY;

  // C11 7.12.7.3 forbids undue overflow or underflow: scale the squares by an
  // exact power of two when they would leave the double range.
  const double ax = fabs(x), ay = fabs(y);
  const double hi = ax > ay ? ax : ay;
  if (hi > 0x1p500)
    return scaled_hypot(ax, ay, 0x1p-600, 0x1p600);
  if (hi < 0x1p-500)
    return scaled_hypot(ax, ay, 0x1p600, 0x1p-600);
  return sqrt(x * x + y * y);
}
