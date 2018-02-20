#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#define fma_def(type, name, isnan_func, isinf_func, signbit_func, fma_func)    \
  type name(type x, type y, type z)                                            \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return fma_func(x, y, z);                                                  \
  }                                                                            \
                                                                               \
  type __##name(type x, type y, type z)                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y, z);                                                      \
  }

fma_def(float, fmaf, isnanf, isinff, __signbitf, __ESBMC_fmaf);
fma_def(double, fma, isnan, isinf, __signbit, __ESBMC_fmad);
fma_def(long double, fmal, isnanl, isinfl, __signbitl, __ESBMC_fmald);

#undef fma_def
