#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#define copysign_def(type, name, signbit_func, isnan_func, abs_func)           \
  type name(type x, type y)                                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    _Bool y_sign = signbit_func(y);                                            \
                                                                               \
    /* Easy case, both signs are equal, just return x */                       \
    if(signbit_func(x) == y_sign)                                              \
      return x;                                                                \
                                                                               \
    if(isnan_func(x))                                                          \
      return y_sign ? -NAN : NAN;                                              \
                                                                               \
    return y_sign ? -abs_func(x) : abs_func(x);                                \
  }                                                                            \
                                                                               \
  type __##name(type x, type y)                                                \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y);                                                         \
  }

copysign_def(float, copysignf, __signbitf, isnanf, fabsf);
copysign_def(double, copysign, __signbit, isnan, fabs);
copysign_def(long double, copysignl, __signbitl, isnanl, fabsl);

#undef copysign_def
