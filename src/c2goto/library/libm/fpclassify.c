#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#define classify_def(type, name, isnan_func, isinf_func, isnormal_func)        \
  int name(type f)                                                             \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return isnan_func(f)                                                       \
             ? FP_NAN                                                          \
             : isinf_func(f)                                                   \
                 ? FP_INFINITE                                                 \
                 : f == 0 ? FP_ZERO                                            \
                          : isnormal_func(f) ? FP_NORMAL : FP_SUBNORMAL;       \
  }

classify_def(float, _fdclass, isnanf, isinff, __ESBMC_isnormalf);
classify_def(float, __fpclassifyf, isnanf, isinff, __ESBMC_isnormalf);

classify_def(double, _dclass, isnan, isinf, __ESBMC_isnormald);
classify_def(double, __fpclassify, isnan, isinf, __ESBMC_isnormald);

classify_def(long double, _ldclass, isnanl, isinfl, __ESBMC_isnormalld);
classify_def(long double, __fpclassifyl, isnanl, isinfl, __ESBMC_isnormalld);

#undef classify_def
