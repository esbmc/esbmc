
#include <assert.h>

int value = 0;
int value2 = 0;

#pragma contract
void increment(void)
{
  __ESBMC_assigns(value);
  
  value = value + 2;  
}

int main(void)
{
  increment();
  __ESBMC_loop_invariant(value2 == 0);
  while (1) {
    increment();
    if(value < 0) {
      break;
    }
  }
  assert(value2 == 0);
  
  return 0;
}
