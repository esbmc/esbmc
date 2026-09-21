/* esbmc/esbmc#7900: --interval-analysis-assume-asserts narrowed the state to
   x > 10 after the first assertion, so the second was folded away although
   --multi-property checks it on the paths where the first one fails. */
#include <assert.h>
int main()
{
  int x = nondet_int();
  assert(x > 10);
  assert(x > 5);
  return 0;
}
