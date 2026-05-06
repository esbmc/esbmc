#include <math.h>

double acosh(double x)
{
__ESBMC_HIDE:;
  return log(x + sqrt(x * x - 1.0));
}
