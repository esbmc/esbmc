#include <stdint.h>
#include <fenv.h>
#include <math.h>
#include <errno.h>
#include <float.h>

#undef errno
extern _Thread_local int errno;

#define NEXTAFTER(suff, dbl, ui)                                               \
  dbl nextafter##suff(dbl x, dbl y)                                            \
  {                                                                            \
    if (isnan(y))                                                              \
      return y;                                                                \
    union                                                                      \
    {                                                                          \
      dbl d;                                                                   \
      ui u;                                                                    \
    } v;                                                                       \
    v.d = x;                                                                   \
    int dir = isless(x, y) ? +1 : isgreater(x, y) ? -1 : 0;                    \
    if (signbit(x))                                                            \
      dir = -dir;                                                              \
    switch (fpclassify(x))                                                     \
    {                                                                          \
    case FP_NAN:                                                               \
      break;                                                                   \
    case FP_ZERO:                                                              \
      if (!dir)                                                                \
        return y;                                                              \
      v.u = 1;                                                                 \
      v.d = copysign##suff(v.d, y);                                            \
      break;                                                                   \
    case FP_INFINITE:                                                          \
    case FP_NORMAL:                                                            \
    case FP_SUBNORMAL:                                                         \
      v.u += dir;                                                              \
      break;                                                                   \
    default:                                                                   \
      __ESBMC_assert(0, "invalid fpclassify value");                           \
    }                                                                          \
    if (isfinite(x) && !isfinite(v.d))                                         \
    {                                                                          \
      feraiseexcept(FE_OVERFLOW);                                              \
      errno = ERANGE;                                                          \
      v.d = copysign##suff(HUGE_VAL, x);                                       \
    }                                                                          \
    if (                                                                       \
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

/* ESBMC encodes a 128-bit long double as binary128 whatever LDBL_MANT_DIG says
 * (frexp.c), so step by the target's ulp instead of on the bit pattern. Every
 * step lands on a representable value, so the sum is exact. */
long double nextafterl(long double x, long double y)
{
  if (isnan(x) || isnan(y))
    return x + y;
  if (x == y)
    return y;
  long double r;
  if (x == 0)
    r = copysignl(LDBL_TRUE_MIN, y);
  else if (isinf(x))
    return copysignl(LDBL_MAX, x);
  else
  {
    int e;
    frexpl(x, &e);
    long double ulp =
      ldexpl(1.0L, (e < LDBL_MIN_EXP ? LDBL_MIN_EXP : e) - LDBL_MANT_DIG);
    /* below a power of two the spacing halves */
    if (
      (x < y) != (x > 0) && e > LDBL_MIN_EXP && fabsl(x) == ldexpl(1.0L, e - 1))
      ulp /= 2;
    r = x < y ? x + ulp : x - ulp;
  }
  if (isinf(r))
  {
    feraiseexcept(FE_OVERFLOW);
    errno = ERANGE;
  }
  else if (!isnormal(r))
  {
    feraiseexcept(FE_UNDERFLOW);
    errno = ERANGE;
  }
  return r;
}

#define NEXTTOWARD(suff, dbl, inf)                                             \
  dbl nexttoward##suff(dbl x, long double y)                                   \
  {                                                                            \
    if (isnan(y) || x == y)                                                    \
      return y;                                                                \
    return nextafter##suff(x, x < y ? inf : -inf);                             \
  }

NEXTTOWARD(, double, HUGE_VAL)
NEXTTOWARD(f, float, HUGE_VALF)

#undef NEXTTOWARD

long double nexttowardl(long double x, long double y)
{
  return nextafterl(x, y);
}
