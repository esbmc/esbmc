/* A loop entered past its head is unwound with its base case run once on
 * entry, so the __ESBMC_old capture keeps its entry value and this true
 * invariant verifies. */
#include <assert.h>
extern int nondet_int(void);
int main()
{
  int i = 0, x = 5;
  if (nondet_int())
    goto mid;
  __ESBMC_loop_invariant(i >= 0 && i <= 10 && x == __ESBMC_old(x) + i);
  while (i < 10)
  {
    i = i + 1;
    x = x + 1;
  mid:;
  }
  assert(x == 15);
  return 0;
}
