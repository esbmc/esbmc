#include <math.h>

double cosh(double x)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  return (exp(x) + exp(-x)) * 0.5;
}
