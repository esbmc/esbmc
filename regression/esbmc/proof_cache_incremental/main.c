#include <assert.h>

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x > 0 && x < 10);
  assert(x + 1 > x);
  assert(x + 1 > x);
  return 0;
}
