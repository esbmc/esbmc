/* #7516: a counterexample against the loop-invariant abstraction leaves the
   base case unknown under a k-step strategy. */
#include <assert.h>
int main()
{
  int i = 0, s = 0;
  __ESBMC_loop_invariant(s == i && i <= 3);
  while (i < 3)
  {
    s++;
    i++;
  }
  assert(s >= 0);
  return 0;
}
