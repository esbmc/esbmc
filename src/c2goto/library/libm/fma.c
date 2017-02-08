#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef fmaf
#undef fma
#undef fmal

float fmaf(float x, float y, float z)
{
  // If x or y are NaN, NaN is returned
  if(__ESBMC_isnanf(x) || __ESBMC_isnanf(y))
    return NAN;

  // If x is zero and y is infinite or if x is infinite and y is zero,
  // and z is not a NaN, a domain error shall occur, and either a NaN,
  // or an implementation-defined value shall be returned.

  // If x is zero and y is infinite or if x is infinite and y is zero,
  // and z is a NaN, then NaN is returned and FE_INVALID may be raised
  if((x == 0.0 && __ESBMC_isinff(y)) || (y == 0.0 && __ESBMC_isinff(x)))
    return NAN;

  // If z is NaN, and x*y aren't 0*Inf or Inf*0, then NaN is returned
  // (without FE_INVALID)
  if(__ESBMC_isnanf(z))
    return NAN;

  // If x*y is an exact infinity and z is an infinity with the opposite sign,
  // NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinff(x*y) && __ESBMC_isinff(z))
    if(__ESBMC_signf(x*y) != __ESBMC_signf(z))
      return NAN;

  return __ESBMC_fmaf(x, y, z);
}

double fma(double x, double y, double z)
{
  // If x or y are NaN, NaN is returned
  if(__ESBMC_isnand(x) || __ESBMC_isnand(y))
    return NAN;

  // If x is zero and y is infinite or if x is infinite and y is zero,
  // and z is not a NaN, a domain error shall occur, and either a NaN,
  // or an implementation-defined value shall be returned.

  // If x is zero and y is infinite or if x is infinite and y is zero,
  // and z is a NaN, then NaN is returned and FE_INVALID may be raised
  if((x == 0.0 && __ESBMC_isinfd(y)) || (y == 0.0 && __ESBMC_isinfd(x)))
    return NAN;

  // If z is NaN, and x*y aren't 0*Inf or Inf*0, then NaN is returned
  // (without FE_INVALID)
  if(__ESBMC_isnand(z))
    return NAN;

  // If x*y is an exact infinity and z is an infinity with the opposite sign,
  // NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinfd(x*y) && __ESBMC_isinfd(z))
    if(__ESBMC_signd(x*y) != __ESBMC_signd(z))
      return NAN;

  return __ESBMC_fmad(x, y, z);
}

long double fmal(long double x, long double y, long double z)
{
  // If x or y are NaN, NaN is returned
  if(__ESBMC_isnanld(x) || __ESBMC_isnanld(y))
    return NAN;

  // If x is zero and y is infinite or if x is infinite and y is zero,
  // and z is not a NaN, a domain error shall occur, and either a NaN,
  // or an implementation-defined value shall be returned.

  // If x is zero and y is infinite or if x is infinite and y is zero,
  // and z is a NaN, then NaN is returned and FE_INVALID may be raised
  if((x == 0.0 && __ESBMC_isinfld(y)) || (y == 0.0 && __ESBMC_isinfld(x)))
    return NAN;

  // If z is NaN, and x*y aren't 0*Inf or Inf*0, then NaN is returned
  // (without FE_INVALID)
  if(__ESBMC_isnanld(z))
    return NAN;

  // If x*y is an exact infinity and z is an infinity with the opposite sign,
  // NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinfld(x*y) && __ESBMC_isinfld(z))
    if(__ESBMC_signld(x*y) != __ESBMC_signld(z))
      return NAN;

  return __ESBMC_fmald(x, y, z);
}
