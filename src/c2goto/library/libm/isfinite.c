#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef isfinite
#undef __finite
#undef __finitef
#undef __finitel

int isfinite(double d) { return __ESBMC_isfinited(d); }

int __finite(double d) { return __ESBMC_isfinited(d); }

int __finitef(float f) { return __ESBMC_isfinitef(f); }

int __finitel(long double ld) { return __ESBMC_isfiniteld(ld); }

