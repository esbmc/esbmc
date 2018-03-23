#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#undef isnan

#define isnan_def(type, name, isnan_func)                                      \
  int name(type f)                                                             \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return isnan_func(f);                                                      \
  }                                                                            \
                                                                               \
  int __##name(type f)                                                         \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

isnan_def(float, isnanf, __ESBMC_isnanf);
isnan_def(double, isnan, __ESBMC_isnand);
isnan_def(long double, isnanl, __ESBMC_isnanld);

#undef isnan_def
