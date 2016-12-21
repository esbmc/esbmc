#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include <fenv.h>
#include "../intrinsics.h"

#undef truncf
#undef trunc
#undef truncl

float truncf(float f)
{
  float result;
  int save_round = fegetround();
  fesetround(FE_TOWARDZERO);
  result = rintf(f);
  fesetround(save_round);
  return result;
}

double trunc(double d)
{
  double result;
  int save_round = fegetround();
  fesetround(FE_TOWARDZERO);
  result = rint(d);
  fesetround(save_round);
  return result;
}

long double truncl(long double ld)
{
  long double result;
  int save_round = fegetround();
  fesetround(FE_TOWARDZERO);
  result = rintl(ld);
  fesetround(save_round);
  return result;
}

