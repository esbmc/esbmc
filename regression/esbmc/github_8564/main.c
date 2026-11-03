#include <assert.h>
#include <stdio.h>

int nondet_int(void);
unsigned nondet_uint(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x >= 100 && x <= 999);
  assert(printf("v=%d\n", x) == 6);

  int n = nondet_int();
  __ESBMC_assume(n > -100 && n <= -10);
  assert(fprintf(stderr, "%i", n) == 3);

  unsigned u = nondet_uint();
  __ESBMC_assume(u >= 10000u && u < 100000u);
  assert(dprintf(1, "[%u]", u) == 7);
  assert(printf("%d", 4294967295u) == 2);
  return 0;
}
