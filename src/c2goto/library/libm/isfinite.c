#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#undef isfinite

#define isfinite_def(type, name, isfinite_func)                                \
  int name(type f)                                                             \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return isfinite_func(f);                                                   \
  }                                                                            \
                                                                               \
  int __##name(type f)                                                         \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

isfinite_def(double, isfinite, __ESBMC_isfinited);

isfinite_def(float, finitef, __ESBMC_isfinitef);
isfinite_def(double, finite, __ESBMC_isfinited);
isfinite_def(long double, finitel, __ESBMC_isfiniteld);

#undef isfinite_def
