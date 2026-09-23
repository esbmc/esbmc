/* --multi-fail-fast skips the second claim at k = 1, once the first is
   violated. The base case at k = 2 solves it, the first being settled, so
   the forward condition's proof at k = 2 reaches it: what k = 1 skipped
   does not count against a later k. */
#include <assert.h>

int main()
{
  unsigned n = nondet_uint();
  unsigned m = nondet_uint();
  assert(n != 7);
  assert(m + 1 > m || m == 0xffffffffu);
  return 0;
}
