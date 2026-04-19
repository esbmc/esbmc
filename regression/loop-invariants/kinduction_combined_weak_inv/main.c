/* Combined --loop-invariant + --k-induction mode: CORRECT but WEAK invariant.
 *
 * The invariant "x >= 0 && y >= 0" is inductive, but does not capture the
 * relationship x == y, so the inductive step cannot close on its own.
 * The base case (bounded unrolling) must still prove the post-loop assertion.
 * Verification should ultimately succeed without any false counterexample.
 */
#include <assert.h>

int main(void)
{
  unsigned int x = 0;
  unsigned int y = x;

  __ESBMC_loop_invariant(x >= 0 && y >= 0);
  while (x < 10)
  {
    x++;
    y++;
  }

  assert(x == y);
  return 0;
}
