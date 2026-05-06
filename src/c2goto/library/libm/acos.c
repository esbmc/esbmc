
#include <math.h>

double acos(double x)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  return atan2(sqrt(1.0 - x * x), x);
}

double arccos(double x)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  return acos(x);
}

double __acos(double x)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  return acos(x);
}
