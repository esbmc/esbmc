/* Regression: GitHub #7494 -- the verification branch copied no body for a
 * do-while, so ASSUME(INV) was followed straight by ASSERT(INV) and any
 * invariant passed.  `x <= 2` is false once the third iteration runs. */
#include <assert.h>

int main()
{
  int x = 0;
  __ESBMC_loop_invariant(x <= 2);
  do
  {
    x++;
  } while (x < 5);
  return 0;
}
