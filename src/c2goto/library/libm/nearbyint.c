#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#define nearbyint_def(type, name, isinf_func, isnan_func, nearbyint_func)      \
  type name(type f)                                                            \
  {                                                                            \
  __ESBMC_HIDE:                                                                \
    if(f == 0.0 || isinf_func(f) || isnan_func(f))                             \
      return f;                                                                \
    return nearbyint_func(f);                                                  \
  }                                                                            \
                                                                               \
  type __##name(type f)                                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

nearbyint_def(float, nearbyintf, isinff, isnanf, __ESBMC_nearbyintf);
nearbyint_def(double, nearbyint, isinf, isnan, __ESBMC_nearbyintd);
nearbyint_def(long double, nearbyintl, isinfl, isnanl, __ESBMC_nearbyintld);

#undef nearbyint_def
