/* FUNCTION: abs */

inline int abs(int i) { return __ESBMC_abs(i); }

/* FUNCTION: labs */

inline long int labs(long int i) { return __ESBMC_labs(i); }

/* FUNCTION: fabs */

inline double fabs(double d) { return __ESBMC_fabs(d); }

/* FUNCTION: fabsl */

inline long double fabsl(long double d) { return __ESBMC_fabsl(d); }

/* FUNCTION: fabsf */

inline float fabsf(float f) { return __ESBMC_fabsf(f); }

/* FUNCTION: isfinite */

int isfinite(double d) { return __ESBMC_isfinite(d); }

/* FUNCTION: isinf */

inline int isinf(double d) { return __ESBMC_isinf(d); }

/* FUNCTION: isnan */

inline int isnan(double d) { return __ESBMC_isnan(d); }

/* FUNCTION: isnormal */

int isnormal(double d) { return __ESBMC_isnormal(d); }

/* FUNCTION: signbit */

inline int signbit(double d) { return __ESBMC_sign(d); }

/* FUNCTION: __fpclassifyd */

#ifndef __ESBMC_MATH_H_INCLUDED
#include <math.h>
#define __ESBMC_MATH_H_INCLUDED
#endif

inline int __fpclassifyd(double d) {
  if(__ESBMC_isnan(d)) return FP_NAN;
  if(__ESBMC_isinf(d)) return FP_INFINITE;
  if(d==0) return FP_ZERO;
  if(__ESBMC_isnormal(d)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

/* FUNCTION: __fpclassifyf */

#ifndef __ESBMC_MATH_H_INCLUDED
#include <math.h>
#define __ESBMC_MATH_H_INCLUDED
#endif

inline int __fpclassifyf(float f) {
  if(__ESBMC_isnan(f)) return FP_NAN;
  if(__ESBMC_isinf(f)) return FP_INFINITE;
  if(f==0) return FP_ZERO;
  if(__ESBMC_isnormal(f)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

/* FUNCTION: __fpclassify */

#ifndef __ESBMC_MATH_H_INCLUDED
#include <math.h>
#define __ESBMC_MATH_H_INCLUDED
#endif

inline int __fpclassify(long double d) {
  if(__ESBMC_isnan(d)) return FP_NAN;
  if(__ESBMC_isinf(d)) return FP_INFINITE;
  if(d==0) return FP_ZERO;
  if(__ESBMC_isnormal(d)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

/* FUNCTION: fegetround */

int fegetround() { return __ESBMC_rounding_mode; }

/* FUNCTION: fesetround */

int fesetround(int __rounding_direction) {
  __ESBMC_rounding_mode=__rounding_direction;
}
