/* Regression: GitHub #7478 -- same defect reached through a guard that calls a
 * function.  The DECL/FUNCTION_CALL pair sits between the loop head and the
 * IF, so the call was evaluated on the pre-havoc state. */
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
  assert(0);
  return 0;
}
