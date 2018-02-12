#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>
#include "../intrinsics.h"

#define fma_def(type, name, isnan_func, isinf_func, signbit_func, fma_func)        \
  type name(type x, type y, type z)                                                \
  {                                                                                \
  __ESBMC_HIDE:;                                                                   \
    /* If x or y are NaN, NaN is returned */                                       \
    if(isnan_func(x) || isnan_func(y))                                             \
      return NAN;                                                                  \
                                                                                   \
    /* If x is zero and y is infinite or if x is infinite and y is zero, */        \
    /* and z is not a NaN, a domain error shall occur, and either a NaN, */        \
    /* or an implementation-defined value shall be returned. */                    \
                                                                                   \
    /* If x is zero and y is infinite or if x is infinite and y is zero, */        \
    /* and z is a NaN, then NaN is returned and FE_INVALID may be raised */        \
    if((x == 0.0 && isinf_func(y)) || (y == 0.0 && isinf_func(x)))                 \
      return NAN;                                                                  \
                                                                                   \
    /* If z is NaN, and x*y aren't 0*Inf or Inf*0, then NaN is returned */         \
    /* (without FE_INVALID) */                                                     \
    if(isnan_func(z))                                                              \
      return NAN;                                                                  \
                                                                                   \
    /* If x*y is an exact infinity and z is an infinity with the opposite sign, */ \
    /* NaN is returned and FE_INVALID is raised */                                 \
    type mult = x * y;                                                             \
    if(isinf_func(mult) && isinf_func(z))                                          \
      if(signbit_func(mult) != signbit_func(z))                                    \
        return NAN;                                                                \
                                                                                   \
    return fma_func(x, y, z);                                                      \
  }                                                                                \
                                                                                   \
  type __##name(type x, type y, type z)                                            \
  {                                                                                \
  __ESBMC_HIDE:;                                                                   \
    return name(x, y, z);                                                          \
  }

fma_def(float, fmaf, isnanf, isinff, __signbitf, __ESBMC_fmaf);
fma_def(double, fma, isnan, isinf, __signbit, __ESBMC_fmad);
fma_def(long double, fmal, isnanl, isinfl, __signbitl, __ESBMC_fmald);

#undef fma_def
