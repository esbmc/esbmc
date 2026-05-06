#include <math.h>

double acosh(double x)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  return log(x + sqrt(x * x - 1.0));
}
