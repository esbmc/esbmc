#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef isnan
#undef __isnan

#undef isnanf
#undef __isnanf

#undef isnanl
#undef __isnanl

inline int isnan(double d) { return __ESBMC_isnand(d); }

inline int __isnan(double d) { return __ESBMC_isnand(d); }

inline int isnanf(float f) { return __ESBMC_isnanf(f); }

inline int __isnanf(float f) { return __ESBMC_isnanf(f); }

inline int isnanl(long double ld) { return __ESBMC_isnanld(ld); }

inline int __isnanl(long double ld) { return __ESBMC_isnanld(ld); }

