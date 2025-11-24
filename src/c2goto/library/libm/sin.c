
#include <math.h>

double sin(double x)
{
__ESBMC_HIDE:;
  return cos(M_PI_2 - x);
}

double __sin(double x)
{
__ESBMC_HIDE:;
  return sin(x - M_PI_2);
}
