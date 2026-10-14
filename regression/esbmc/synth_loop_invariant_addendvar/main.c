/* The accumulator's addend is the counter, which the loop modifies. The
 * closed form s0 + (i - i0) * e is only valid for an e that is constant across
 * the loop, so the recogniser declines. */
#include <assert.h>

int main(void)
{
  unsigned int n;
  __ESBMC_assume(n <= 4);

  unsigned int i = 0;
  unsigned int s = 0;

  while (i < n)
  {
    s = s + i;
    i = i + 1;
  }

  assert(s <= 6);
  return 0;
}
