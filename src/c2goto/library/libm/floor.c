#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include <fenv.h>

#define floor_def(type, name, rint_func)                                       \
  type name(type f)                                                            \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    type result;                                                               \
    int save_round = fegetround();                                             \
    fesetround(FE_DOWNWARD);                                                   \
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

floor_def(float, floorf, rintf);
floor_def(double, floor, rint);
floor_def(long double, floorl, rintl);

#undef floor_def
