
#include <fenv.h>
#include <math.h>

#define modff_def(type, name, nearbyint_func, copysign_func, isinf_func)       \
  type name(type value, type *iptr)                                            \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    int save_round = fegetround();                                             \
    fesetround(FE_TOWARDZERO);                                                 \
    *iptr = nearbyint_func(value);                                             \
    fesetround(save_round);                                                    \
    return copysign_func(isinf_func(value) ? 0.0 : value - (*iptr), value);    \
  }                                                                            \
                                                                               \
  type __##name(type value, type *iptr)                                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return name(value, iptr);                                                  \
  }

modff_def(float, modff, nearbyintf, copysignf, isinf);
modff_def(double, modf, nearbyint, copysign, isinf);
modff_def(long double, modfl, nearbyintl, copysignl, isinf);

#undef modff_def
