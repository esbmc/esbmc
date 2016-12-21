#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef remainderf
#undef remainder
#undef remainderl

float remainderf(float x, float y)
{
  int quo;
  return remquof(x, y, &quo);
}

double remainder(double x, double y)
{
  int quo;
  return remquo(x, y, &quo);
}

long double remainderl(long double x, long double y)
{
  int quo;
  return remquol(x, y, &quo);
}

