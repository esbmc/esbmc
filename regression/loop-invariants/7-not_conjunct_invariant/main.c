#include <assert.h>

int main()
{
  int i = 0;
  int sum = 0;

  __ESBMC_loop_invariant(i >= 0 && i <= 5000000 && sum == i * 10);
  int j =
    0; //should pass? should fail? Currently it is not supported and would cause continue unwinding
  while (i < 5000000)
  {
    sum += 10;
    i++;
  }
  assert(sum == 50000000);
  return 0;
}