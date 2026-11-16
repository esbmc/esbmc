#include <math.h>

double asinh(double x)
{
__ESBMC_HIDE:;
  // C23 F.10.2.2: asinh(±0) is ±0 and asinh(±inf) is ±inf; NaN propagates.
  // Tiny arguments return x, as log(1 + tiny) would round to 0.
  double a = fabs(x);
  if (isnan(x) || isinf(x) || a < 0x1p-28)
    return x;
  // asinh is odd (C23 F.10p3), so work on |x|: x + sqrt(x * x + 1) cancels to
  // 0 for large negative x. The three ranges follow glibc's s_asinh.c: past
  // 2^28 a * a can overflow and sqrt(a * a + 1) is a, and below 2 log1p keeps
  // the precision log loses near 1.
  double r = a > 0x1p28 ? log(a) + M_LN2
             : a > 2.0  ? log(2.0 * a + 1.0 / (sqrt(a * a + 1.0) + a))
                        : log1p(a + a * a / (1.0 + sqrt(1.0 + a * a)));
  return signbit(x) ? -r : r;
}
