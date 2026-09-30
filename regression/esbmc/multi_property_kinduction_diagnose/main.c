/* The same program without --multi-fail-fast: the base case at k = 2 solves
   every claim, so the diagnostic inductive step at max-k proves the other
   two. */
#include <assert.h>
_Bool nondet_bool();
unsigned nondet_uint();
int main()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n < 10);
  unsigned i = 0, j = 3;
  while (nondet_bool())
  {
    i++;
    assert(i != 2);
    assert(n < 10);
    assert(j == 3);
  }
  return 0;
}
