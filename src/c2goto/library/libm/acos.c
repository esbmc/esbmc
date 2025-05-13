
#include <math.h>

double acos(double x)
{
__ESBMC_HIDE:;
  return atan2(sqrt(1.0 - x * x), x);
}

double __acos(double x)
{
__ESBMC_HIDE:;
  return acos(x);
}
