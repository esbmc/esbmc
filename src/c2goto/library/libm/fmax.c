#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#define fmax_def(type, name, isnan_func)                                       \
  type name(type x, type y)                                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    int x_is_nan = isnan_func(x);                                              \
    int y_is_nan = isnan_func(y);                                              \
                                                                               \
    /* If both argument are NaN, NaN is returned */                            \
    if(x_is_nan && y_is_nan)                                                   \
      return NAN;                                                              \
                                                                               \
    /* If one arg is NaN, the other is returned */                             \
    if(x_is_nan)                                                               \
      return y;                                                                \
                                                                               \
    if(y_is_nan)                                                               \
      return x;                                                                \
                                                                               \
    return (x > y ? x : y);                                                    \
  }                                                                            \
                                                                               \
  type __##name(type x, type y)                                                \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y);                                                         \
  }

fmax_def(float, fmaxf, isnan);
fmax_def(double, fmax, isnan);
fmax_def(long double, fmaxl, isnan);

#undef fmax_def
