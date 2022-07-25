#include <stdint.h>
#include <fenv.h>
#include <math.h>
#include <errno.h>

#undef errno
extern _Thread_local int errno;

#define NEXTAFTER(suff, dbl, ui)                                               \
  dbl nextafter##suff(dbl x, dbl y)                                            \
  {                                                                            \
    if(isnan(y))                                                               \
      return y;                                                                \
    union                                                                      \
    {                                                                          \
      dbl d;                                                                   \
      ui u;                                                                    \
    } v;                                                                       \
    v.d = x;                                                                   \
    int dir = isless(x, y) ? +1 : isgreater(x, y) ? -1 : 0;                    \
    if(signbit(x))                                                             \
      dir = -dir;                                                              \
    switch(fpclassify(x))                                                      \
    {                                                                          \
    case FP_NAN:                                                               \
      break;                                                                   \
    case FP_ZERO:                                                              \
      if(!dir)                                                                 \
        return y;                                                              \
      v.u = 1;                                                                 \
      v.d = copysign##suff(v.d, y);                                            \
      break;                                                                   \
    case FP_INFINITE:                                                          \
      break;                                                                   \
    case FP_NORMAL:                                                            \
    case FP_SUBNORMAL:                                                         \
      v.u += dir;                                                              \
      break;                                                                   \
    default:                                                                   \
      __ESBMC_assert(0, "invalid fpclassify value");                           \
    }                                                                          \
    if(isfinite(x) && !isfinite(v.d))                                          \
    {                                                                          \
      feraiseexcept(FE_OVERFLOW);                                              \
      errno = ERANGE;                                                          \
      v.d = copysign##suff(HUGE_VAL, x);                                       \
    }                                                                          \
    if(                                                                        \
      islessgreater(x, y) && isfinite(v.d) &&                                  \
      (!isnormal(v.d) || v.d == (dbl)0))                                       \
    {                                                                          \
      feraiseexcept(FE_UNDERFLOW);                                             \
      errno = ERANGE;                                                          \
    }                                                                          \
    return v.d;                                                                \
  }

NEXTAFTER(, double, uint64_t)
NEXTAFTER(f, float, uint32_t)

#undef NEXTAFTER
