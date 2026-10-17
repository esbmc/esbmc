#include <math.h>
#include <limits.h>

/* C11 7.12.6.5 and 7.12.6.11: the unbiased exponent of x, which frexp gives
 * for the range [0.5, 1), one more than for logb's range [1, 2). */
#define LOGB(type, suff)                                                       \
  int ilogb##suff(type x)                                                      \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    if (x == 0)                                                                \
      return FP_ILOGB0;                                                        \
    if (isnan(x))                                                              \
      return FP_ILOGBNAN;                                                      \
    if (isinf(x))                                                              \
      return INT_MAX;                                                          \
    int e;                                                                     \
    frexp##suff(x, &e);                                                        \
    return e - 1;                                                              \
  }                                                                            \
                                                                               \
  type logb##suff(type x)                                                      \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    if (x == 0)                                                                \
      return -HUGE_VAL;                                                        \
    if (!isfinite(x))                                                          \
      return x * x;                                                            \
    int e;                                                                     \
    frexp##suff(x, &e);                                                        \
    return e - 1;                                                              \
  }

LOGB(float, f)
LOGB(double, )
LOGB(long double, l)

#undef LOGB
