#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef isinf
#undef __isinf

#undef isinff
#undef __isinff

#undef isinfl
#undef __isinfl

inline int isinf(double d) { return __ESBMC_isinfd(d); }

inline int __isinf(double d) { return __ESBMC_isinfd(d); }

inline int isinff(float f) { return __ESBMC_isinff(f); }

inline int __isinff(float f) { return __ESBMC_isinff(f); }

inline int isinfl(long double ld) { return __ESBMC_isinfld(ld); }

inline int __isinfl(long double ld) { return __ESBMC_isinfld(ld); }

