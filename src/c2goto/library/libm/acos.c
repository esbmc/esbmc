
#include <math.h>

double acos(double x)
{
__ESBMC_HIDE:;
  return 1 / cos(x);
}

double __acos(double x)
{
__ESBMC_HIDE:;
  return acos(x);
}
