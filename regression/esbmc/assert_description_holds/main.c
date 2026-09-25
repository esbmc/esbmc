#include <assert.h>

int nondet_int();

int main()
{
  int x = nondet_int();
  __ESBMC_assume(x > 5);
  assert(x > 5);
  return 0;
}
