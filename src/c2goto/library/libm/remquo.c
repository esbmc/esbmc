#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include <fenv.h>

#define remquo_def(type, name, isnan_func, isinf_func, llrint_func)                    \
  type name(type x, type y, int *quo)                                                  \
  {                                                                                    \
  __ESBMC_HIDE:;                                                                       \
    /* If either argument is NaN, NaN is returned */                                   \
    if(isnan_func(x) || isnan_func(y))                                                 \
      return NAN;                                                                      \
                                                                                       \
    /* If y is +0.0/-0.0 and x is not NaN, NaN is returned and FE_INVALID is raised */ \
    if(y == 0.0)                                                                       \
      return NAN;                                                                      \
                                                                                       \
    /* If x is +inf/-inf and y is not NaN, NaN is returned and FE_INVALID is raised */ \
    if(__ESBMC_isinff(x))                                                              \
      return NAN;                                                                      \
                                                                                       \
    /* If y is +inf/-inf, return x */                                                  \
    if(isinf_func(y))                                                                  \
      return x;                                                                        \
                                                                                       \
    /* remainder = x - rquot * y */                                                    \
    /* Where rquot is the result of: x/y, rounded toward the nearest */                \
    /* integral value (with halfway cases rounded toward the even number). */          \
                                                                                       \
    /* Save previous rounding mode */                                                  \
    int old_rm = fegetround();                                                         \
                                                                                       \
    /* Set round to nearest */                                                         \
    fesetround(FE_TONEAREST);                                                          \
                                                                                       \
    /* Perform division */                                                             \
    long long rquot = llrint_func(x / y);                                              \
                                                                                       \
    /* Restore old rounding mode */                                                    \
    fesetround(old_rm);                                                                \
                                                                                       \
    return x - (y * rquot);                                                            \
  }                                                                                    \
                                                                                       \
  type __##name(type x, type y, int *quo)                                              \
  {                                                                                    \
  __ESBMC_HIDE:;                                                                       \
    return name(x, y, quo);                                                            \
  }

remquo_def(float, remquof, isnanf, isinff, llrintf);
remquo_def(double, remquo, isnan, isinf, llrint);
remquo_def(long double, remquol, isnanl, isinfl, llrintl);

#undef remquo_def
