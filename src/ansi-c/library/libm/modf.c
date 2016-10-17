#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include <fenv.h>
#include "../intrinsics.h"

#undef modff
#undef modf
#undef modfl

float modff(float value, float* iptr)
{
  int save_round = fegetround();
  fesetround(FE_TOWARDZERO);
  *iptr = nearbyint(value);
  fesetround(save_round);
  return copysign(isinf(value) ? 0.0 : value - (*iptr), value);
}

double modf(double value, double* iptr)
{
  int save_round = fegetround();
  fesetround(FE_TOWARDZERO);
  *iptr = nearbyint(value);
  fesetround(save_round);
  return copysign(isinf(value) ? 0.0 : value - (*iptr), value);
}

long double modfl(long double value, long double* iptr)
{
  int save_round = fegetround();
  fesetround(FE_TOWARDZERO);
  *iptr = nearbyint(value);
  fesetround(save_round);
  return copysign(isinf(value) ? 0.0 : value - (*iptr), value);
}

