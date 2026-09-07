/* Regression: GitHub #7478 -- the failing half.  The post-loop assertion is
 * plainly false; before the fix it was never reached and the run reported
 * VERIFICATION SUCCESSFUL. */
#include <assert.h>

int main()
{
  int cnt = 4;
  __ESBMC_loop_invariant(cnt >= 0);
  while (cnt--)
  {
  }
  assert(0);
  return 0;
}
