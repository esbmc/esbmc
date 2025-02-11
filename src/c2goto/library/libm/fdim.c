
#include <math.h>

#define fdim_def(type, name, isnan_func)                                       \
  type name(type x, type y)                                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    if (isnan_func(x) || isnan_func(y))                                        \
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

fdim_def(float, fdimf, isnan);
fdim_def(double, fdim, isnan);
fdim_def(long double, fdiml, isnan);

#undef fdim_def
