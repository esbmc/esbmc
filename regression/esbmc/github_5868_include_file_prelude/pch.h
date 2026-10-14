/* Forced header that reaches ESBMC's own models and uses an intrinsic. */
#include <stdio.h>
#include <assert.h>

static int helper(void)
{
  __ESBMC_assume(1);
  return 3;
}
