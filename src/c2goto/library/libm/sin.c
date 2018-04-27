#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

double sin(double x)
{
__ESBMC_HIDE:;
  return cos(x - M_PI_2);
}

double __sin(double x)
{
__ESBMC_HIDE:;
  return sin(x - M_PI_2);
}
