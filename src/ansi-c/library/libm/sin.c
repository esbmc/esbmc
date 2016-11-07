#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef sin

double sin(double x)
{
  return cos(x - M_PI_2);
}

