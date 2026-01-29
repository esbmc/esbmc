
#include <assert.h>

#pragma contract
void increment(int *x)
{
  __ESBMC_requires(*x >= 0);
  __ESBMC_ensures(*x == __ESBMC_old(*x) + 2);
  __ESBMC_assigns(*x);
  
  *x = *x + 2;  
}

int main(void)
{
  int value = 0;  
  __ESBMC_loop_invariant(value >= 0);
  while (1) {
    increment(&value);
  }
  assert(value >= 0);
  
  return 0;
}
