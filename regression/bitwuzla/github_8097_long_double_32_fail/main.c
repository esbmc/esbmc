#include <assert.h>

long double nondet_ld(void);

int main(void)
{
  /* Under --32, long double is 96 bits wide, a format Bitwuzla cannot build. */
  long double x = nondet_ld();
  __ESBMC_assume(x > 1.0L && x < 2.0L);
  long double y = x * 2.0L;
  assert(y > 2.0L && y < 3.0L);
  return 0;
}
