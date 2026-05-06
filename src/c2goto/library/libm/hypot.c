#include <math.h>

double hypot(double x, double y)
{
// cppcheck-suppress unusedLabel
__ESBMC_HIDE:;
  return sqrt(x * x + y * y);
}
