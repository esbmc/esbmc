#include <math.h>

double asin(double x)
{
__ESBMC_HIDE:;
  if (x > 1.0 || x < -1.0)
    return NAN;
  return atan2(x, sqrt(1.0 - x * x));
}

double arcsin(double x)
{
__ESBMC_HIDE:;
  return asin(x);
}

double __asin(double x)
{
__ESBMC_HIDE:;
  return asin(x);
}
