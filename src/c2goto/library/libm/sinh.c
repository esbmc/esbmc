#include <math.h>

double sinh(double x)
{
__ESBMC_HIDE:;
  return (exp(x) - exp(-x)) * 0.5;
}
