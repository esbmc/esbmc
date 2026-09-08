/* Regression: GitHub #7494 -- the combined pass killed a do-while's exit path.
 * `x <= 4` holds at every loop head; the loop leaves x == 5, so this assertion
 * is false and must be reported. */
#include <assert.h>

int main()
{
  int x = 0;
  __ESBMC_loop_invariant(x <= 4);
  do
  {
    x++;
  } while (x < 5);
  assert(x == 4);
  return 0;
}
