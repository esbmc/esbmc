/* Regression: GitHub #7480 -- the scoping control.  This assertion runs before
 * any havoc, so it is checked against the concrete program and its
 * counterexample is reproducible.  It must stay FAILED even though the same
 * function carries an annotated loop. */
#include <assert.h>

int main()
{
  int i = 0, s = 0;
  assert(s == 3);
  __ESBMC_loop_invariant(i >= 0);
  while (i < 3)
  {
    s++;
    i++;
  }
  return 0;
}
