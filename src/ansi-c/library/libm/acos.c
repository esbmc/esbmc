#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef acos

double acos(double x)
{
  return 1/cos(x);
}

