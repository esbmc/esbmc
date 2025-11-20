
#include <assert.h>

int main()
{
  int i = 0;
  int sum = 0;
  int secret = 42;

  /*@ loop invariant i >= 0 && i <= 5000000 && sum == i * 10;
     @*/

  __ESBMC_loop_invariant(i >= 0 && i <= 5000000 && sum == i * 10);
  for (i = 0; i < 5000000; i++)
  {
    sum += 10;
    secret = secret + 1;
  }

  assert(sum == 50000000);
  //assert(secret == 5000042); // 42+5000000
  return 0;
}
