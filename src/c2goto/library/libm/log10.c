#include <math.h>

double log10(double x)
{
__ESBMC_HIDE:;
  return log(x) / M_LN10;
}
