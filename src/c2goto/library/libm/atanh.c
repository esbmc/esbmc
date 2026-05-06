#include <math.h>

double atanh(double x)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  return 0.5 * log((1.0 + x) / (1.0 - x));
}
