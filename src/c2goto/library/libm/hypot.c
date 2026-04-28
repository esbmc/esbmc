#include <math.h>

double hypot(double x, double y)
{
__ESBMC_HIDE:;
  return sqrt(x * x + y * y);
}
