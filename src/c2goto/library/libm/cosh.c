#include <math.h>

double cosh(double x)
{
__ESBMC_HIDE:;
  return (exp(x) + exp(-x)) * 0.5;
}
