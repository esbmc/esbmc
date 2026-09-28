// Forced header that reaches ESBMC's stream models (which use nondet_*) and
// uses an ESBMC intrinsic directly.
#include <iostream>
#include <cassert>

static int helper()
{
  __ESBMC_assume(1);
  return 3;
}
