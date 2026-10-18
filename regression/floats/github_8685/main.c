/* github #8685: ldexp rounds subnormal and overflowing results, C11 7.12.6.6 */
#include <assert.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>

int nondet_int(void);

int main(void)
{
  double tiny = 0x1p-1074;
  assert(ldexp(tiny, -1) == 0.0);
  assert(ldexp(tiny, 1) == 0x1p-1073);
  assert(ldexp(tiny, 1074) == 1.0);
  assert(ldexp(1.5, -1074) == 0x1p-1073);
  assert(ldexp(0x1.8p-1022, -1) == 0x1.8p-1023);
  assert(ldexp(1.0, 1024) == INFINITY);
  assert(ldexp(1.0, INT_MAX) == INFINITY);
  assert(ldexp(-1.0, INT_MIN) == 0.0 && signbit(ldexp(-1.0, INT_MIN)));
  assert(ldexp(0x1.fffffffffffffp-1, -1022) == 0x1p-1022);
  assert(ldexpf(0x1p-149f, 1) == 0x1p-148f);
  assert(ldexpf(0x1.8p-149f, -1) == 0x1p-149f);
  assert(ldexpl(0x1p-16494L, -1) == 0.0L);
  assert(ldexpl(0x1.8p-16494L, 1) == 0x1.8p-16493L);
  assert(scalbn(0x1.8p-1070, -4) == 0x1p-1073);

  fesetround(FE_UPWARD);
  assert(ldexp(-1.0, 1024) == -DBL_MAX);
  assert(ldexp(1.0, -2000) == tiny);
  fesetround(FE_DOWNWARD);
  assert(ldexp(-1.0, -2000) == -tiny);
  assert(ldexp(1.0, 1024) == DBL_MAX);
  fesetround(FE_TOWARDZERO);
  assert(ldexp(-1.0, 1024) == -DBL_MAX);
  assert(ldexp(0x1.fp-1070, -4) == tiny);
  fesetround(FE_TONEAREST);

  int n = nondet_int();
  __ESBMC_assume(n >= -1074 && n <= 1023);
  assert(ldexp(ldexp(1.0, n), -n) == 1.0);
  return 0;
}
