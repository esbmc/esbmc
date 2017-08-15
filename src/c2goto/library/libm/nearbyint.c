#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef nearbyintf
#undef nearbyint
#undef nearbyintl

float nearbyintf(float f)
{
  if(f == 0.0)
    return f;
  if(__ESBMC_isinff(f))
    return f;
  if(__ESBMC_isnanf(f))
    return NAN;
  return __ESBMC_nearbyintf(f);
}

double nearbyint(double d)
{
  if(d == 0.0)
    return d;
  if(__ESBMC_isinfd(d))
    return d;
  if(__ESBMC_isnand(d))
    return NAN;
  return __ESBMC_nearbyintd(d);
}

long double nearbyintl(long double ld)
{
  if(ld == 0.0)
    return ld;
  if(__ESBMC_isinfld(ld))
    return ld;
  if(__ESBMC_isnanld(ld))
    return NAN;
  return __ESBMC_nearbyintld(ld);
}

