/* Counter enters at 2 under a `<` guard. The two-disjunct bound is not
 * establishable for that entry value (it needs the third disjunct i == i0),
 * so the recogniser declines instead of emitting a claim that fails at
 * establishment on a correct program. */
#include <assert.h>

int main(void)
{
  unsigned int n;
  __ESBMC_assume(n <= 6);

  unsigned int i = 2;
  unsigned int s = 0;

  while (i < n)
  {
    s = s + 1;
    i = i + 1;
  }

  assert(s <= 4);
  return 0;
}
