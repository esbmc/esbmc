#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#define fmin_def(type, name, isnan_func)                                       \
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
    return (x < y ? x : y);                                                    \
  }                                                                            \
                                                                               \
  type __##name(type x, type y)                                                \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y);                                                         \
  }

fmin_def(float, fminf, isnanf);
fmin_def(double, fmin, isnan);
fmin_def(long double, fminl, isnanl);

#undef fmin_def
