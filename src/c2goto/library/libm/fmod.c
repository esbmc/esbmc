#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef fmod
#undef fmodl
#undef fmodf

double fmod(double x, double y)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnand(x) || __ESBMC_isnand(y))
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinfd(x))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +0.0/-0.0 and y is not zero, +0.0/-0.0 is returned
  if((x == 0.0) && (y != 0.0))
  {
    if(__ESBMC_signd(x))
      return -0.0;
    else
      return +0.0;
  }

  // If y is +inf/-inf and x is finite, x is returned.
  if(__ESBMC_isinfd(y) && __ESBMC_isfinited(x))
    return x;

  return x - (y * (int)(x/y));
}

float fmodf(float x, float y)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnanf(x) || __ESBMC_isnanf(y))
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinff(x))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +0.0/-0.0 and y is not zero, +0.0/-0.0 is returned
  if((x == 0.0) && (y != 0.0))
  {
    if(__ESBMC_signf(x))
      return -0.0;
    else
      return +0.0;
  }

  // If y is +inf/-inf and x is finite, x is returned.
  if(__ESBMC_isinff(y) && __ESBMC_isfinitef(x))
    return x;

  return x - (y * (int)(x/y));
}

long double fmodl(long double x, long double y)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnanld(x) || __ESBMC_isnanld(y))
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinfld(x))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +0.0/-0.0 and y is not zero, +0.0/-0.0 is returned
  if((x == 0.0) && (y != 0.0))
  {
    if(__ESBMC_signld(x))
      return -0.0;
    else
      return +0.0;
  }

  // If y is +inf/-inf and x is finite, x is returned.
  if(__ESBMC_isinfld(y) && __ESBMC_isfiniteld(x))
    return x;

  return x - (y * (int)(x/y));
}

