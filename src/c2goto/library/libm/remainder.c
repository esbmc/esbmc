#include <fenv.h>
#include <math.h>

/* remainder() calls are normally intercepted by the frontend and lowered to
 * the solver's exact IEEE 754 remainder (fp.rem) -- see ieee_rem in
 * clang_c_adjust_expr.cpp. This body is the fallback for any path that does
 * not intercept, and is deliberately self-contained (no remquo/fmod calls)
 * so the fmod -> remainder and remquo -> remainder chains cannot recurse. */
#define remainder_def(type, name, isnan_func, isinf_func, llrint_func)         \
  type name(type x, type y)                                                    \
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
      return x;                                                                \
                                                                               \
    int old_rm = fegetround();                                                 \
    fesetround(FE_TONEAREST);                                                  \
    long long rquot = llrint_func(x / y);                                      \
    fesetround(old_rm);                                                        \
                                                                               \
    return x - (y * rquot);                                                    \
  }                                                                            \
                                                                               \
  type __##name(type x, type y)                                                \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y);                                                         \
  }

remainder_def(float, remainderf, isnan, isinf, llrintf);
remainder_def(double, remainder, isnan, isinf, llrint);
remainder_def(long double, remainderl, isnan, isinf, llrintl);

#undef remainder_def
