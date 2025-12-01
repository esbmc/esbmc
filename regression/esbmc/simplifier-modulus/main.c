#include <assert.h>

int main()
{
  int x = nondet_int();
  __ESBMC_assume(x!=0);
  assert(x % x == 0);
  return 0;
}
