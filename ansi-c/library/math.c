#define __CRT__NO_INLINE /* Don't let mingw insert code */
#include <math.h>
#undef fpclassify
#undef isfinite
#undef isfinite
#undef isnormal
#undef isnan
#undef isinf
#undef signbit

#include "intrinsics.h"

#ifdef _WIN32
#undef fabs
#undef fabsl
#undef fabsf

// Whipped out of glibc headers. Don't exactly know how these work, but they're
// what the linux version works upon.
#ifdef _MSVC
enum
  {
    FP_NAN,
# define FP_NAN FP_NAN
    FP_INFINITE,
# define FP_INFINITE FP_INFINITE
    FP_ZERO,
# define FP_ZERO FP_ZERO
    FP_SUBNORMAL,
# define FP_SUBNORMAL FP_SUBNORMAL
    FP_NORMAL
# define FP_NORMAL FP_NORMAL
  };
#endif
#endif

int abs(int i) { return __ESBMC_abs(i); }
long int labs(long int i) { return __ESBMC_labs(i); }
double fabs(double d) { return __ESBMC_fabs(d); }
long double fabsl(long double d) { return __ESBMC_fabsl(d); }
float fabsf(float f) { return __ESBMC_fabsf(f); }
int isfinite(double d) { return __ESBMC_isfinite(d); }
int isinf(double d) { return __ESBMC_isinf(d); }
int isnan(double d) { return __ESBMC_isnan(d); }
int isnormal(double d) { return __ESBMC_isnormal(d); }
int signbit(double d) { return __ESBMC_sign(d); }

int __fpclassifyd(double d) {
  if(__ESBMC_isnan(d)) return FP_NAN;
  if(__ESBMC_isinf(d)) return FP_INFINITE;
  if(d==0) return FP_ZERO;
  if(__ESBMC_isnormal(d)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

int __fpclassifyf(float f) {
  if(__ESBMC_isnan(f)) return FP_NAN;
  if(__ESBMC_isinf(f)) return FP_INFINITE;
  if(f==0) return FP_ZERO;
  if(__ESBMC_isnormal(f)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

int __fpclassify(long double d) {
  if(__ESBMC_isnan(d)) return FP_NAN;
  if(__ESBMC_isinf(d)) return FP_INFINITE;
  if(d==0) return FP_ZERO;
  if(__ESBMC_isnormal(d)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

int fegetround() { return __ESBMC_rounding_mode; }

int fesetround(int __rounding_direction) {
  __ESBMC_rounding_mode=__rounding_direction;
}
