/* --loop-invariant copies the loop body, so assert(n != 7) is two claims.
   --multi-fail-fast skips the second copy at k = 1 once the first fails;
   the base case at k = 2 has to solve it rather than drop it with its
   twin. */
#include <assert.h>

unsigned nondet_uint();

int main()
{
  unsigned n = nondet_uint();
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= 10);
  while (i < 10)
  {
    assert(n != 7);
    assert(i < 10);
    i++;
  }
  return 0;
}
