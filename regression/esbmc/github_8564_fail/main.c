#include <assert.h>
#include <stdio.h>

int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x >= 99 && x <= 999);
  int r = printf("%d", x);
  assert(r == 3);
  return 0;
}
