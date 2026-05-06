#include <math.h>

double exp2(double x)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  return exp(x * M_LN2);
}
