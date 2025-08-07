#include <assert.h>

int main()
{
  int i = 0;
  int sum = 0;
  while (i < 5000000)
  {
    __ESBMC_loop_invariant(i >= 0 && i <= 5000000 && sum == i * 10);
    //should time out since the format is wrong but need to change to continue collecting invariants
    sum += 10;
    i++;
  }
  assert(sum == 50000000);
  return 0;
}