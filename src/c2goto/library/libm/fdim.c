#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#define fdim_def(type, name, isnan_func)                                       \
  type name(type x, type y)                                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    if(isnan_func(x) || isnan_func(y))                                         \
      return NAN;                                                              \
                                                                               \
    return (x > y ? x - y : 0.0);                                              \
  }                                                                            \
                                                                               \
  type __##name(type x, type y)                                                \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y);                                                         \
  }

fdim_def(float, fdimf, isnanf);
fdim_def(double, fdim, isnan);
fdim_def(long double, fdiml, isnanl);

#undef fdim_def
