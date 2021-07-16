#include <assert.h>

int main()
{
  int x=nondet_int();
  while(x>0)
  {
   __ESBMC_assume(x>100);
   x--;
  }
  assert(0);
  return 0;
}
