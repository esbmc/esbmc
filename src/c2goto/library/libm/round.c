#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include <fenv.h>

#define round_def(ret_type, type, name, rint_func, copysign_func, abs_func)    \
  ret_type name(type f)                                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    type result;                                                               \
    int save_round = fegetround();                                             \
    fesetround(FE_TOWARDZERO);                                                 \
    result = rint_func(copysign_func(0.5 + abs_func(f), f));                   \
    fesetround(save_round);                                                    \
    return result;                                                             \
  }                                                                            \
                                                                               \
  ret_type __##name(type f)                                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

round_def(float, float, roundf, rintf, copysignf, fabsf);
round_def(double, double, round, rint, copysign, fabs);
round_def(long double, long double, roundl, rintl, copysignl, fabsl);

round_def(long, float, lroundf, rintf, copysignf, fabsf);
round_def(long, double, lround, rint, copysign, fabs);
round_def(long, long double, lroundl, rintl, copysignl, fabsl);

round_def(long long, float, llroundf, rintf, copysignf, fabsf);
round_def(long long, double, llround, rint, copysign, fabs);
round_def(long long, long double, llroundl, rintl, copysignl, fabsl);

#undef round_def
