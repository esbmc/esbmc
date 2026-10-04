#include <assert.h>

/* #8102: UINT_MAX + 1U wraps to 0. */
int main(void)
{
  unsigned int x = __VERIFIER_nondet_uint();
  __ESBMC_assume(x == 4294967295U);
  if (x + 1U != 0)
    assert(0);
  return 0;
}
