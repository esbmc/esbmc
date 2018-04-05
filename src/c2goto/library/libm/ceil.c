#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include <fenv.h>

#define ceil_def(type, name, rint_func)                                        \
  type name(type f)                                                            \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    type result;                                                               \
    int save_round = fegetround();                                             \
    fesetround(FE_UPWARD);                                                     \
    result = rint_func(f);                                                     \
    fesetround(save_round);                                                    \
    return result;                                                             \
  }                                                                            \
                                                                               \
  type __##name(type f)                                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

ceil_def(float, ceilf, rintf);
ceil_def(double, ceil, rint);
ceil_def(long double, ceill, rintl);

#undef ceil_def
