#include <math.h>

double tan(double x)
{
__ESBMC_HIDE:;
  return sin(x) / cos(x);
}
