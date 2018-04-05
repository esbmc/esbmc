#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#undef isinf

#define isinf_def(type, name, isinf_func)                                      \
  int name(type f)                                                             \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return isinf_func(f);                                                      \
  }                                                                            \
                                                                               \
  int __##name(type f)                                                         \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

isinf_def(float, isinff, __ESBMC_isinff);
isinf_def(double, isinf, __ESBMC_isinfd);
isinf_def(long double, isinfl, __ESBMC_isinfld);

#undef isinf_def
