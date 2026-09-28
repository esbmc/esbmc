/* Regression: GitHub #7480 -- the passing half.  Strengthening the invariant
 * so it carries s == 3 out of the loop discharges the same assertion, which
 * shows the downgrade to UNKNOWN follows from the abstraction being too coarse
 * rather than from every post-havoc claim being given up on. */
#include <assert.h>

int main()
{
  int i = 0, s = 0;
  __ESBMC_loop_invariant(i >= 0 && i <= 3 && s == i);
  while (i < 3)
  {
    s++;
    i++;
  }
  assert(s == 3);
  return 0;
}
