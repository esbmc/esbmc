/* The assertion follows from the invariant alone, so the inductive step closes
 * at k = 2 and the cost does not grow with the loop bound.  A bound of 100000
 * is far beyond what bounded unrolling could reach within the k-step limit. */
#include <assert.h>

int main(void)
{
  unsigned int x = 0;
  unsigned int y = 0;

  __ESBMC_loop_invariant(x == y);
  while (x < 100000)
  {
    x++;
    y++;
  }

  assert(x == y);
  return 0;
}
