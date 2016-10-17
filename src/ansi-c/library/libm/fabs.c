#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef fabs
#undef fabsl
#undef fabsf

double fabs(double d)
{
  if(d == 0.0)
    return 0.0;
  if(__ESBMC_isinfd(d))
    return INFINITY;
  if(__ESBMC_isnand(d))
    return NAN;
  return __ESBMC_fabsd(d);
}

long double fabsl(long double ld)
{
  if(ld == 0.0)
    return 0.0;
  if(__ESBMC_isinfld(ld))
    return INFINITY;
  if(__ESBMC_isnanld(ld))
    return NAN;
  return __ESBMC_fabsld(ld);
}

float fabsf(float f)
{
  if(f == 0.0)
    return 0.0;
  if(__ESBMC_isinff(f))
    return INFINITY;
  if(__ESBMC_isnanf(f))
    return NAN;
  return __ESBMC_fabsf(f);
}

