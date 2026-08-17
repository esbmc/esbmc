#include <fenv.h>
#include <math.h>

/* remquo: the remainder is exactly what remainder() returns -- normally the
 * solver's fp.rem via frontend interception, with remainder.c's
 * self-contained body as the fallback (that body never calls back here, so
 * the chain cannot recurse).
 *
 * *quo receives the sign and low bits of the rounded-to-nearest integral
 * quotient x/y (C17 7.12.10.3 requires at least three bits). It is computed
 * with llrint(x/y), which double-rounds and can overflow long long for huge
 * quotient magnitudes; C only defines *quo modulo a power of two, so the
 * low bits remain meaningful in the common range. */
#define remquo_def(                                                            \
  type, name, isnan_func, isinf_func, llrint_func, remainder_func)             \
  type name(type x, type y, int *quo)                                          \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    if (isnan_func(x) || isnan_func(y))                                        \
      return NAN;                                                              \
                                                                               \
    if (y == 0.0)                                                              \
      return NAN;                                                              \
                                                                               \
    if (isinf_func(x))                                                         \
      return NAN;                                                              \
                                                                               \
    if (isinf_func(y))                                                         \
    {                                                                          \
      *quo = 0;                                                                \
      return x;                                                                \
    }                                                                          \
                                                                               \
    int old_rm = fegetround();                                                 \
    fesetround(FE_TONEAREST);                                                  \
    long long rquot = llrint_func(x / y);                                      \
    fesetround(old_rm);                                                        \
                                                                               \
    *quo = (int)rquot;                                                         \
    return remainder_func(x, y);                                               \
  }                                                                            \
                                                                               \
  type __##name(type x, type y, int *quo)                                      \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y, quo);                                                    \
  }

remquo_def(float, remquof, isnan, isinf, llrintf, remainderf);
remquo_def(double, remquo, isnan, isinf, llrint, remainder);
remquo_def(long double, remquol, isnan, isinf, llrintl, remainderl);

#undef remquo_def
