#include <math.h>

/* Moves x's exponent towards double's normal range by 3k, so that the
 * conversion to double keeps it and the scale's cube root, 2^k, is exact. */
#define CBRTL_SCALE(pow3k, threshold, powk)                                    \
  if (fabsl(x) > 0x1p999L && fabsl(x) >= threshold)                            \
  {                                                                            \
    x /= pow3k;                                                                \
    scale *= powk;                                                             \
  }                                                                            \
  else if (fabsl(x) < 0x1p-999L && fabsl(x) < 1 / threshold)                   \
  {                                                                            \
    x *= pow3k;                                                                \
    scale /= powk;                                                             \
  }

long double cbrtl(long double x)
{
__ESBMC_HIDE:;
  if (x == 0 || !isfinite(x))
    return x;
  long double scale = 1;
  CBRTL_SCALE(0x1p12288L, 0x1p11289L, 0x1p4096L)
  CBRTL_SCALE(0x1p6144L, 0x1p5145L, 0x1p2048L)
  CBRTL_SCALE(0x1p3072L, 0x1p2073L, 0x1p1024L)
  CBRTL_SCALE(0x1p1536L, 0x1p537L, 0x1p512L)
  CBRTL_SCALE(0x1p768L, 0x1p-231L, 0x1p256L)
  /* Each Newton step doubles the correct bits of cbrt's 53, which binary128's
   * 113 need twice. */
  long double y = cbrt((double)x);
  y -= (y * y * y - x) / (3 * y * y);
  y -= (y * y * y - x) / (3 * y * y);
  return y * scale;
}

#undef CBRTL_SCALE
