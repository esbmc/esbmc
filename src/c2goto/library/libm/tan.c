#include <math.h>

double tan(double x)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  return sin(x) / cos(x);
}
