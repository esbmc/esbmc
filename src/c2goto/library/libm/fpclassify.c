#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#undef _dclass
#undef _ldclass
#undef _fdclass

#undef __fpclassifyd
#undef __fpclassifyf
#undef __fpclassify
#undef __fpclassifyl

inline short _dclass(double d) {
  __ESBMC_HIDE:;
  return __ESBMC_isnand(d)?FP_NAN:
         __ESBMC_isinfd(d)?FP_INFINITE:
         d==0?FP_ZERO:
         __ESBMC_isnormald(d)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline short _ldclass(long double ld) {
  __ESBMC_HIDE:;
  return __ESBMC_isnanld(ld)?FP_NAN:
         __ESBMC_isinfld(ld)?FP_INFINITE:
         ld==0?FP_ZERO:
         __ESBMC_isnormalld(ld)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline short _fdclass(float f) {
  __ESBMC_HIDE:;
  return __ESBMC_isnanf(f)?FP_NAN:
         __ESBMC_isinff(f)?FP_INFINITE:
         f==0?FP_ZERO:
         __ESBMC_isnormalf(f)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline int __fpclassifyd(double d) {
  __ESBMC_HIDE:;
  return __ESBMC_isnand(d)?FP_NAN:
         __ESBMC_isinfd(d)?FP_INFINITE:
         d==0?FP_ZERO:
         __ESBMC_isnormald(d)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline int __fpclassifyl(long double f) {
  __ESBMC_HIDE:;
  return __ESBMC_isnanld(f)?FP_NAN:
         __ESBMC_isinfld(f)?FP_INFINITE:
         f==0?FP_ZERO:
         __ESBMC_isnormalld(f)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline int __fpclassify(double d) {
  __ESBMC_HIDE:;
  return __ESBMC_isnand(d)?FP_NAN:
         __ESBMC_isinfd(d)?FP_INFINITE:
         d==0?FP_ZERO:
         __ESBMC_isnormald(d)?FP_NORMAL:
         FP_SUBNORMAL;
}

inline int __fpclassifyf(float f)
{
  __ESBMC_HIDE:;
  return __ESBMC_isnanf(f)?FP_NAN:
         __ESBMC_isinff(f)?FP_INFINITE:
         f==0?FP_ZERO:
         __ESBMC_isnormalf(f)?FP_NORMAL:
         FP_SUBNORMAL;
}

