#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include <fenv.h>
#include "../intrinsics.h"

#undef remquof
#undef remquo
#undef remquol

float remquof(float x, float y, int *quo)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnanf(x) || __ESBMC_isnanf(y))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinff(x))
    return NAN;

  // If y is +inf/-inf, return x
  if(__ESBMC_isinff(y))
    return x;

  // remainder = x - rquot * y
  // Where rquot is the result of: x/y, rounded toward the nearest
  // integral value (with halfway cases rounded toward the even number).

  // Save previous rounding mode
  int old_rm = fegetround();

  // Set round to nearest
  fesetround(FE_TONEAREST);

  // Perform division
  long long rquot = llrintf(x/y);

  // Restore old rounding mode
  fesetround(old_rm);

  return x - (y * rquot);
}

double remquo(double x, double y, int *quo)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnand(x) || __ESBMC_isnand(y))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinfd(x))
    return NAN;

  // If y is +inf/-inf, return x
  if(__ESBMC_isinfd(y))
    return x;

  // remainder = x - rquot * y
  // Where rquot is the result of: x/y, rounded toward the nearest
  // integral value (with halfway cases rounded toward the even number).

  // Save previous rounding mode
  int old_rm = fegetround();

  // Set round to nearest
  fesetround(FE_TONEAREST);

  // Perform division
  long long rquot = llrint(x/y);

  // Restore old rounding mode
  fesetround(old_rm);

  return x - (y * rquot);
}

long double remquol(long double x, long double y, int *quo)
{
  // If either argument is NaN, NaN is returned
  if(__ESBMC_isnanld(x) || __ESBMC_isnanld(y))
    return NAN;

  // If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised
  if(y == 0.0)
    return NAN;

  // If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised
  if(__ESBMC_isinfld(x))
    return NAN;

  // If y is +inf/-inf, return x
  if(__ESBMC_isinfld(y))
    return x;

  // remainder = x - rquot * y
  // Where rquot is the result of: x/y, rounded toward the nearest
  // integral value (with halfway cases rounded toward the even number).

  // Save previous rounding mode
  int old_rm = fegetround();

  // Set round to nearest
  fesetround(FE_TONEAREST);

  // Perform division
  long long rquot = llrintl(x/y);

  // Restore old rounding mode
  fesetround(old_rm);

  return x - (y * rquot);
}

