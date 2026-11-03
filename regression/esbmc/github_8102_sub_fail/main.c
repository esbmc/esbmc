#include <assert.h>

/* #8102: 0U - 1U wraps to UINT_MAX. */
int main(void)
{
  unsigned int x = __VERIFIER_nondet_uint();
  __ESBMC_assume(x == 0);
  if (x - 1U == 4294967295U)
    assert(0);
  return 0;
}
