#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include <fenv.h>
#include "../intrinsics.h"

#undef roundf
#undef round
#undef roundl

#undef lroundf
#undef lround
#undef lroundl

#undef llroundf
#undef llround
#undef llroundl

float roundf(float f)
{
  float result;
  int save_round = fegetround();
  fesetround(FE_TOWARDZERO);
  result = rintf(copysign(0.5 + fabsf(f), f));
  fesetround(save_round);
  return result;
}

double round(double d)
{
  double result;
  int save_round = fegetround();
  fesetround(FE_TOWARDZERO);
  result = rint(copysign(0.5 + fabs(d), d));
  fesetround(save_round);
  return result;
}

long double roundl(long double ld)
{
  long double result;
  int save_round = fegetround();
  fesetround(FE_TOWARDZERO);
  result = rintl(copysign(0.5 + fabsl(ld), ld));
  fesetround(save_round);
  return result;
}

long lroundf(float f) { return roundf(f); }

long lround(double d) { return round(d); }

long lroundl(long double ld) { return roundl(ld); }

long long llroundf(float f) { return roundf(f); }

long long llround(double d) { return round(d); }

long long llroundl(long double ld) { return roundl(ld); }

