/* Regression: GitHub #7585 -- the negative control. The invariant is true but
 * leaves the claim open: the abstraction admits states where it holds and
 * states where it does not, so the violation is not the invariant's and the
 * verdict stays unknown, as #7480 requires. */
#include <assert.h>

int main(void)
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
