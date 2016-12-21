#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include <fenv.h>

#include "../intrinsics.h"

#undef floorf
#undef floor
#undef floorl

float floorf(float f)
{
  float result;
  int save_round = fegetround();
  fesetround(FE_DOWNWARD);
  result = rintf(f);
  fesetround(save_round);
  return result;
}

double floor(double d)
{
  double result;
  int save_round = fegetround();
  fesetround(FE_DOWNWARD);
  result = rint(d);
  fesetround(save_round);
  return result;
}

long double floorl(long double ld)
{
  long double result;
  int save_round = fegetround();
  fesetround(FE_DOWNWARD);
  result = rintl(ld);
  fesetround(save_round);
  return result;
}

