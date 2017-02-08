#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef isnormal
#undef __isnormalf
#undef __isnormald
#undef __isnormall

inline int isnormal(double d) { return __ESBMC_isnormald(d); }

inline int __isnormalf(float f) { return __ESBMC_isnormalf(f); }

inline int __isnormald(double d) { return __ESBMC_isnormald(d); }

inline int __isnormall(long double ld) { return __ESBMC_isnormalld(ld); }

