/* Regression: GitHub #7478 -- the passing half for a guard that calls a
 * function.  The exit state has to be constrained, not merely reachable: the
 * loop leaves `cnt == 0`, which only holds if the guard is re-evaluated on the
 * havoc'd state. */
#include <assert.h>

static int positive(int c)
{
  return c > 0;
}

int main()
{
  int cnt = 4;
  __ESBMC_loop_invariant(cnt >= 0);
  while (positive(cnt))
  {
    cnt--;
  }
  assert(cnt == 0);
  return 0;
}
