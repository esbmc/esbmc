#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef copysignf
#undef copysign
#undef copysignl

float copysignf(float x, float y)
{
  // Easy case, both signs are equal, just return x
  if(__ESBMC_signf(x) == __ESBMC_signf(y))
    return x;

  _Bool y_sign = __ESBMC_signf(y);

  if(__ESBMC_isnanf(x))
    return y_sign ? -NAN : NAN;

  return y_sign ? -__ESBMC_fabsf(x) : __ESBMC_fabsf(x);
}

double copysign(double x, double y)
{
  // Easy case, both signs are equal, just return x
  if(__ESBMC_signd(x) == __ESBMC_signd(y))
    return x;

  _Bool y_sign = __ESBMC_signd(y);

  if(__ESBMC_isnand(x))
    return y_sign ? -NAN : NAN;

  return y_sign ? -__ESBMC_fabsd(x) : __ESBMC_fabsd(x);
}

long double copysignl(long double x, long double y)
{
  // Easy case, both signs are equal, just return x
  if(__ESBMC_signld(x) == __ESBMC_signld(y))
    return x;

  _Bool y_sign = __ESBMC_signld(y);

  if(__ESBMC_isnanld(x))
    return y_sign ? -NAN : NAN;

  return y_sign ? -__ESBMC_fabsld(x) : __ESBMC_fabsld(x);
}

