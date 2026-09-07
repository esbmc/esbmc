/* Regression: GitHub #7478 -- the failing half.  With the write invisible to
 * the havoc the loop stayed concrete, its exit edge was infeasible and this
 * assertion was reported as passing. */
#include <assert.h>

int main()
{
  int cnt = 4;
  int *p = &cnt;
  __ESBMC_loop_invariant(cnt >= 0);
  while (*p)
  {
    (*p)--;
  }
  assert(0);
  return 0;
}
