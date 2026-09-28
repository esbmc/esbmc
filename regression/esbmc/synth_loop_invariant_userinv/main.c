/* The loop carries a user-written invariant and is otherwise exactly the shape
 * the recogniser accepts. The user's invariant is authoritative: a synthesised
 * marker would be folded into the same conjunction by the extractor, so a
 * rejected guess would fail the user's own proof. Synthesis must stand down
 * and the user's proof must still go through. */
#include <assert.h>

int main(void)
{
  unsigned int n;
  __ESBMC_assume(n <= 4);

  unsigned int i = 0;

  __ESBMC_loop_invariant(i <= n);
  while (i < n)
    i++;

  assert(i == n || n == 0);
  return 0;
}
