/* Regression: GitHub #7478 -- the passing half for a short-circuit guard.
 * `a--` runs before the test, so the loop leaves `a == -1`. */
#include <assert.h>

int main()
{
  int a = 4, b = 1;
  __ESBMC_loop_invariant(a >= 0);
  while (a-- && b)
  {
  }
  assert(a == -1);
  return 0;
}
