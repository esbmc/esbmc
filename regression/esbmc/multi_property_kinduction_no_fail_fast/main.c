/* The same program without --multi-fail-fast: every claim is solved at every
   base case, so the forward condition's proof reaches the second one even
   though the first was violated at k = 1. */
#include <assert.h>

int main()
{
  unsigned n = nondet_uint();
  unsigned m = nondet_uint();
  assert(n != 7);
  assert(m + 1 > m || m == 0xffffffffu);
  return 0;
}
