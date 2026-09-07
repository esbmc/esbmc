/* Regression: GitHub #7478 -- a loop guard with a side effect is lowered to
 * instructions that sit between the loop head and the guard's IF.  Havoc'ing
 * after them left the IF testing a pre-havoc temporary, so the loop-exit edge
 * was infeasible and every claim after the loop was dropped unsolved. */
#include <assert.h>

int main()
{
  int cnt = 4;
  __ESBMC_loop_invariant(cnt >= 0);
  while (cnt--)
  {
  }
  assert(cnt == -1);
  return 0;
}
