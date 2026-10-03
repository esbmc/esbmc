#include <assert.h>

/* #8096: negating INT_MIN wraps back to INT_MIN (--overflow-check is off). */
int main(void)
{
  int s = __VERIFIER_nondet_int();
  __ESBMC_assume(s == -2147483647 - 1);
  int r = -s;
  if (r < 0)
    assert(0);
  return 0;
}
