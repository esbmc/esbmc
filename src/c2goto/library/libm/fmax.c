#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef fmaxf
#undef fmax
#undef fmaxl

float fmaxf(float x, float y)
{
  if(__ESBMC_isnanf(x) || __ESBMC_isnanf(y))
  {
    // If both argument are NaN, NaN is returned
    if(__ESBMC_isnanf(x) && __ESBMC_isnanf(y))
      return NAN;

    // Otherwise, return the side that is not NaN
    if(__ESBMC_isnanf(x))
      return y;

    return x;
  }

  return (x > y ? x : y);
}

double fmax(double x, double y)
{
  if(__ESBMC_isnand(x) || __ESBMC_isnand(y))
  {
    // If both argument are NaN, NaN is returned
    if(__ESBMC_isnand(x) && __ESBMC_isnand(y))
      return NAN;

    // Otherwise, return the side that is not NaN
    if(__ESBMC_isnand(x))
      return y;

    return x;
  }

  return (x > y ? x : y);
}

long double fmaxl(long double x, long double y)
{
  if(__ESBMC_isnanld(x) || __ESBMC_isnanld(y))
  {
    // If both argument are NaN, NaN is returned
    if(__ESBMC_isnanld(x) && __ESBMC_isnanld(y))
      return NAN;

    // Otherwise, return the side that is not NaN
    if(__ESBMC_isnanld(x))
      return y;

    return x;
  }

  return (x > y ? x : y);
}
