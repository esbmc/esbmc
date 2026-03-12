/* Combined --loop-invariant + --k-induction mode: WRONG invariant.
 *
 * The invariant "y >= 100" is false at loop entry (y starts at 0),
 * so Branch 1 should catch it immediately as a base-case violation.
 */
#include <assert.h>

int main(void)
{
  unsigned int x = 0;
  unsigned int y = x;

  __ESBMC_loop_invariant(x >= 0 && y >= 100);
  while (x < 10)
  {
    x++;
    y++;
  }

  assert(x == y);
  return 0;
}
