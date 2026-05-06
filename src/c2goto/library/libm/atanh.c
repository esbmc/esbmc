#include <math.h>

double atanh(double x)
{
__ESBMC_HIDE:;
  return 0.5 * log((1.0 + x) / (1.0 - x));
}
