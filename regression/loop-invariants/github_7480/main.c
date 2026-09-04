/* Regression: GitHub #7480 -- the invariant is genuine but too weak to carry
 * the post-loop assertion.  The claim is checked against the havoc'd state, so
 * its counterexample (i == 3, s == 0) is unreachable in the program and the
 * verdict must be "cannot prove", not "the program is wrong". */
#include <assert.h>

int main()
{
  int i = 0, s = 0;
  __ESBMC_loop_invariant(i >= 0);
  while (i < 3)
  {
    s++;
    i++;
  }
  assert(s == 3);
  return 0;
}
