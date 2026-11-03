#include <assert.h>

/* #8096: -1U is 4294967295U, so the branch is never taken. */
int main(void)
{
  unsigned int x = __VERIFIER_nondet_uint();
  __ESBMC_assume(x == 4294967295U);
  if (x != -1U)
    assert(0);
  return 0;
}
