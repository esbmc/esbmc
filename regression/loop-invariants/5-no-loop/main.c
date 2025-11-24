

#include <assert.h>

int main()
{
  int i = 0;
  int sum = 0;

  __ESBMC_loop_invariant(sum == 50000000);
  assert(sum == 50000000);
  return 0;
}