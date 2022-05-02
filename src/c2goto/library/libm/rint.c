#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#define rint_def(ret_type, type, name, isnan_func)                             \
  ret_type name(type f)                                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return isnan_func(f);                                                      \
  }                                                                            \
                                                                               \
  ret_type __##name(type f)                                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

rint_def(float, float, rintf, nearbyintf);
rint_def(double, double, rint, nearbyint);
rint_def(long double, long double, rintl, nearbyintl);

rint_def(long, float, lrintf, nearbyintf);
rint_def(long, double, lrint, nearbyint);
rint_def(long, long double, lrintl, nearbyintl);

rint_def(long long, float, llrintf, nearbyintf);
rint_def(long long, double, llrint, nearbyint);
rint_def(long long, long double, llrintl, nearbyintl);

#undef rint_def
