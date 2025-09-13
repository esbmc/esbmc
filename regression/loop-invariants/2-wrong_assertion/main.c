
#include <assert.h>

int main()
{
  int i = 0;
  int sum = 0;

  /*@ loop invariant i >= 0 && i < 5000000 && sum == i * 10; overflow should fail 
       @*/

  __ESBMC_loop_invariant(i >= 0 && i <= 5000000 && sum == i * 10);
  while (i < 5000000) //replace the condition to i <5000000
  {
    sum += 10;
    i++;
  }
  assert(sum == 0);
  return 0;
}