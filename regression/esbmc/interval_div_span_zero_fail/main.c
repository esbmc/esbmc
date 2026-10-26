#include <assert.h>

/* A divisor range spanning 0 reaches the quotients of x == 1 and x == -1, not
 * only those of its endpoints. */
int main(void)
{
  int x = __VERIFIER_nondet_int();
  __ESBMC_assume(x >= -2 && x <= 2 && x != 0);
  int y = 100 / x;
  if (y > 50)
    assert(0);
  return 0;
}
