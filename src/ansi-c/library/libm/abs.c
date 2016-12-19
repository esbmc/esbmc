#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef abs
#undef labs
#undef llabs

int abs(int i) { return __ESBMC_abs(i); }

long labs(long i) { return __ESBMC_labs(i); }

long long llabs(long long i) { return __ESBMC_llabs(i); }

