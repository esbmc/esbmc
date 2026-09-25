/* The same claims under plain BMC, where the unwinding bound is the whole
   analysis: there a per-claim UNSAT is a proof and stays one. */
#include <assert.h>

int main()
{
  unsigned n = nondet_uint();
  unsigned x = 0;
  __ESBMC_assume(n < 5);
  for (unsigned i = 0; i < n; ++i)
    ++x;
  assert(x == n);
  assert(x < 5);
}
