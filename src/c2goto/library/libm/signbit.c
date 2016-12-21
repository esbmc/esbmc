#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef signbit

#undef __signbit
#undef __signbitd
#undef __signbitf
#undef __signbitl

#undef _dsign
#undef _ldsign
#undef _fdsign

inline int signbit(double d) { return __ESBMC_signd(d); }

inline int __signbit(double d) { return __ESBMC_signd(d); }

inline int __signbitd(double d) { return __ESBMC_signd(d); }

inline int __signbitf(float f) { return __ESBMC_signf(f); }

inline int __signbitl(long double ld) { return __ESBMC_signld(ld); }

inline int _dsign(double d) { return __ESBMC_signd(d); }

inline int _ldsign(long double ld) { return __ESBMC_signld(ld); }

inline int _fdsign(float f) { return __ESBMC_signf(f); }

