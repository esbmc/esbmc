
#include <math.h>

#define copysign_def(type, name, signbit_func, abs_func)                       \
  type name(type x, type y)                                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    type abs = abs_func(x);                                                    \
    return (signbit_func(y)) ? -abs : abs;                                     \
  }                                                                            \
                                                                               \
  type __##name(type x, type y)                                                \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(x, y);                                                         \
  }

copysign_def(float, copysignf, signbit, fabsf);
copysign_def(double, copysign, signbit, fabs);
copysign_def(long double, copysignl, signbit, fabsl);

#undef copysign_def
