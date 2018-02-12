#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#define fabs_def(type, name, isinf_func, isnan_func, abs_func)                 \
  type name(type f)                                                            \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    if(f == 0.0)                                                               \
      return 0.0;                                                              \
    if(isinf_func(f))                                                          \
      return INFINITY;                                                         \
    if(isnan_func(f))                                                          \
      return NAN;                                                              \
    return abs_func(f);                                                        \
  }                                                                            \
                                                                               \
  type __##name(type f)                                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

fabs_def(float, fabsf, isinff, isnanf, __ESBMC_fabsf);
fabs_def(double, fabs, isinf, isnan, __ESBMC_fabsd);
fabs_def(long double, fabsl, isinfl, isnanl, __ESBMC_fabsld);

#undef fabs_def
