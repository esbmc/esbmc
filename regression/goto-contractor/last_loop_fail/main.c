#include <assert.h>
int main()
{
  int x = 1;
  int y = 0;

  while(y<10)
  {
    y++;
  }

  __ESBMC_assume(x>0);
  while(x < 2147483640 && __VERIFIER_nondet_int())
  {
    x = x + 1;
  }
  assert(x>10);

  return 0;
}