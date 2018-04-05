#define __CRT__NO_INLINE /* Don't let mingw insert code */

#include <math.h>

#define fabs_def(type, name, isinf_func, isnan_func, abs_func)                 \
  type name(type f)                                                            \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return abs_func(f);                                                        \
  }                                                                            \
                                                                               \
  type __##name(type f)                                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(f);                                                            \
  }

fabs_def(float, fabsf, isinff, isnanf, __ESBMC_fabsf);
fabs_def(double, fabs, isinf, isnan, __ESBMC_fabsd);
fabs_def(long double, fabsl, isinfl, isnanl, __ESBMC_fabsld);

#undef fabs_def
