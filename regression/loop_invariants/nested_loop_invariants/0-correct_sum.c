

#include <assert.h>

int main()
{
  int i = 0;
  int sum = 0;
  int j = 0;

  __ESBMC_loop_invariant(i >= 0 && i <= 10 && sum == i * 10);
  while (i < 10)
  {
    sum += 10;
    i++;
    __ESBMC_loop_invariant(j >= 0 && j <= 10 && sum == i * 10 + j);
    while (j < 10)
    {
        sum++;
        j++;
    }
  }

  assert(sum == 110);
  return 0;
}
