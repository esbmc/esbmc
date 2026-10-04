#include <assert.h>

/* #8102: 65536 * 65536 wraps to 0 in a 32-bit int (--overflow-check is off). */
int main(void)
{
  int s = __VERIFIER_nondet_int();
  __ESBMC_assume(s == 65536);
  if (s * s == 0)
    assert(0);
  return 0;
}
