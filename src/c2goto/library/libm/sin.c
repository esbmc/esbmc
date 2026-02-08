
#include <math.h>

double sin(double x)
{
__ESBMC_HIDE:;
  if (x == 0.0) return 0.0;
  if (x == M_PI) return 0.0;
  return cos(M_PI_2 - x);
}

double __sin(double x)
{
__ESBMC_HIDE:;
  return sin(x - M_PI_2);
}
