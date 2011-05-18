#include <math.h>

inline int abs(int i) { return __ESBMC_abs(i); }
inline long int labs(long int i) { return __ESBMC_labs(i); }
inline double fabs(double d) { return __ESBMC_fabs(d); }
inline long double fabsl(long double d) { return __ESBMC_fabsl(d); }
inline float fabsf(float f) { return __ESBMC_fabsf(f); }
int isfinite(double d) { return __ESBMC_isfinite(d); }
inline int isinf(double d) { return __ESBMC_isinf(d); }
inline int isnan(double d) { return __ESBMC_isnan(d); }
int isnormal(double d) { return __ESBMC_isnormal(d); }
inline int signbit(double d) { return __ESBMC_sign(d); }

inline int __fpclassifyd(double d) {
  if(__ESBMC_isnan(d)) return FP_NAN;
  if(__ESBMC_isinf(d)) return FP_INFINITE;
  if(d==0) return FP_ZERO;
  if(__ESBMC_isnormal(d)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

inline int __fpclassifyf(float f) {
  if(__ESBMC_isnan(f)) return FP_NAN;
  if(__ESBMC_isinf(f)) return FP_INFINITE;
  if(f==0) return FP_ZERO;
  if(__ESBMC_isnormal(f)) return FP_NORMAL;
  return FP_SUBNORMAL;
}

inline int __fpclassify(long double d) {
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
