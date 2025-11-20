

#include <assert.h>

int main()
{
  int i = 0;
  int sum = 0;
  int j = 0;

  __ESBMC_loop_invariant(i >= 0 && i <= 10 && sum == i * 11);
  while (i < 10)
  {
    j = 0;
    
    __ESBMC_loop_invariant(j >= 0 && j <= 10 && sum == i * 11 + j);
    while (j < 10)
    {
      sum++;
      j++;
    }
    
    sum++;  // Add one more to make total 11 per outer iteration
    i++;
  }

  assert(sum == 110);
  return 0;
}
