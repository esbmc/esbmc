
#include <math.h>

double acos(double x)
{
__ESBMC_HIDE:;
  return atan2(sqrt(1.0 - x * x), x);
<<<<<<< HEAD
}

double arccos(double x)
{
__ESBMC_HIDE:;
  return acos(x);
=======
>>>>>>> 50ffb5a8b ([libm] fix acos function (#2442))
}

double arccos(double x)
{
__ESBMC_HIDE:;
  return acos(x);
}

double __acos(double x)
{
__ESBMC_HIDE:;
  return acos(x);
}
