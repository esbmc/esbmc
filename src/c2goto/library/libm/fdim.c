#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef fdimf
#undef fdim
#undef fdiml

float fdimf(float x, float y)
{
  if(__ESBMC_isnanf(x) || __ESBMC_isnanf(y))
    return NAN;

  return (x > y ? x - y : 0.0);
}

double fdim(double x, double y)
{
  if(__ESBMC_isnand(x) || __ESBMC_isnand(y))
    return NAN;

  return (x > y ? x - y : 0.0);
}

long double fdiml(long double x, long double y)
{
  if(__ESBMC_isnanld(x) || __ESBMC_isnanld(y))
    return NAN;

  return (x > y ? x - y : 0.0);
}
