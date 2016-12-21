#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef nanf
#undef nan
#undef nanl

float nanf(const char* arg)
{
  (void) arg;
  return NAN;
}

double nan(const char* arg)
{
  (void) arg;
  return NAN;
}

long double nanl(const char* arg)
{
  (void) arg;
  return NAN;
}
