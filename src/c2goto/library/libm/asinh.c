#include <math.h>

double asinh(double x)
{
__ESBMC_HIDE:;
  return log(x + sqrt(x * x + 1.0));
}
