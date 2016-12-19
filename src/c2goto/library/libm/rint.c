#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef rintf
#undef rint
#undef rintl

#undef lrintf
#undef lrint
#undef lrintl

#undef llrintf
#undef llrint
#undef llrintl

float rintf(float f) { return nearbyintf(f); }

double rint(double d) { return nearbyint(d); }

long double rintl(long double ld) { return nearbyintl(ld); }

long lrintf(float f) { return nearbyintf(f); }

long lrint(double d)  { return nearbyint(d); }

long lrintl(long double ld) { return nearbyintl(ld); }

long long llrintf(float f) { return nearbyintf(f); }

long long llrint(double d)  { return nearbyint(d); }

long long llrintl(long double ld) { return nearbyintl(ld); }

