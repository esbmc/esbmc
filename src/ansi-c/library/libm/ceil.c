#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include <fenv.h>
#include "../intrinsics.h"

#undef ceilf
#undef ceil
#undef ceill

float ceilf(float f)
{
  float result;
  int save_round = fegetround();
  fesetround(FE_UPWARD);
  result = rintf(f);
  fesetround(save_round);
  return result;
}

double ceil(double d)
{
  double result;
  int save_round = fegetround();
  fesetround(FE_UPWARD);
  result = rint(d);
  fesetround(save_round);
  return result;
}

long double ceill(long double ld)
{
  long double result;
  int save_round = fegetround();
  fesetround(FE_UPWARD);
  result = rintl(ld);
  fesetround(save_round);
  return result;
}

