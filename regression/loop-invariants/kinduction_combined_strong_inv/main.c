/* Combined --loop-invariant + --k-induction mode: CORRECT and STRONG invariant.
 *
 * The invariant "x == y" precisely captures the relationship between x and y.
 * Branch 1 verifies inductivity; k-induction can then prove the post-loop
 * assertion using the base case at a small k.
 */
#include <assert.h>

int main(void)
{
  unsigned int x = 0;
  unsigned int y = x;

  __ESBMC_loop_invariant(x == y);
  while (x < 10)
  {
    x++;
    y++;
  }

  assert(x == y);
  return 0;
}
