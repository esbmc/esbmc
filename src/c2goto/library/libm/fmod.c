#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#define fmod_def(type, name, isnan_func, isinf_func, isfinite_func)                    \
  type name(type x, type y)                                                            \
  {                                                                                    \
  __ESBMC_HIDE:;                                                                       \
    int x_is_nan = isnan_func(x);                                                      \
    int y_is_nan = isnan_func(y);                                                      \
                                                                                       \
    /* If either argument is NaN, NaN is returned */                                   \
    if(x_is_nan || y_is_nan)                                                           \
      return NAN;                                                                      \
                                                                                       \
    /* If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised */ \
    if(isinf_func(x))                                                                  \
      return NAN;                                                                      \
                                                                                       \
    /* If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised */ \
    if(y == 0.0)                                                                       \
      return NAN;                                                                      \
                                                                                       \
    /* If x is +0.0/-0.0 and y is not zero, +0.0/-0.0 is returned */                   \
    if(x == 0.0)                                                                       \
      return x;                                                                        \
                                                                                       \
    /* If y is +inf/-inf and x is finite, x is returned. */                            \
    if(isinf_func(y) && isfinite_func(x))                                              \
      return x;                                                                        \
                                                                                       \
    return x - (y * (int)(x / y));                                                     \
  }                                                                                    \
                                                                                       \
  type __##name(type x, type y)                                                        \
  {                                                                                    \
  __ESBMC_HIDE:;                                                                       \
    return name(x, y);                                                                 \
  }

fmod_def(float, fmodf, isnanf, isinff, finitef);
fmod_def(double, fmod, isnan, isinf, finite);
fmod_def(long double, fmodl, isnanl, isinfl, finitel);

#undef fmod_def
