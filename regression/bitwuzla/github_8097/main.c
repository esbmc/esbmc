/* Issue #8097: under --32 long double was a 96-bit (15, 81) float, a format
 * Bitwuzla only builds when compiled with --fpexp, so it aborted. long double
 * is now binary128 there, as under --64. x is nondet so the long double sort
 * reaches the solver. */
#include <assert.h>

long double nondet_ld(void);

int main()
{
  long double x = nondet_ld();
  __ESBMC_assume(x == 1.0L / 3);
  /* 1/3 has no exact double, so the narrowing loses precision. */
  assert((double)x != x);
  return 0;
}
