/* The base case at k = 2 finds the first claim violated in the second
   iteration, and --multi-fail-fast skips the other claims' instances there.
   The diagnostic inductive step at max-k proves both inductive -- the third
   by the simplifier alone -- but that round did not back either proof. */
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
