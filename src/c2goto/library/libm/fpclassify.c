#define __CRT__NO_INLINE /* Don't let mingw insert code */

#ifdef _MSVC
#define _USE_MATH_DEFINES
#define _CRT_FUNCTIONS_REQUIRED 0

// There is no easy way to fix this :(
#define _fdclass crt_fdclass
#define _ldclass crt_ldclass
#define _dclass crt_dclass
#endif
#include <math.h>

#ifdef _MSVC
#undef _fdclass
#undef _ldclass
#undef _dclass
#endif

#ifdef _MSVC

#endif
#define classify_def(type, name, isnan_func, isinf_func, isnormal_func)        \
  int name(type f)                                                             \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return isnan_func(f)      ? FP_NAN                                         \
           : isinf_func(f)    ? FP_INFINITE                                    \
           : f == 0           ? FP_ZERO                                        \
           : isnormal_func(f) ? FP_NORMAL                                      \
                              : FP_SUBNORMAL;                                  \
  }

classify_def(float, _fdclass, isnan, isinf, __builtin_isnormal);
classify_def(float, __fpclassifyf, isnan, isinf, __builtin_isnormal);

classify_def(double, _dclass, isnan, isinf, __builtin_isnormal);
classify_def(double, __fpclassify, isnan, isinf, __builtin_isnormal);
classify_def(double, __fpclassifyd, isnan, isinf, __builtin_isnormal);

classify_def(long double, _ldclass, isnan, isinf, __builtin_isnormal);
classify_def(long double, __fpclassifyl, isnan, isinf, __builtin_isnormal);

#undef classify_def
