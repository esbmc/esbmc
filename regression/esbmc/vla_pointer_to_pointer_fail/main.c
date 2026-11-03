#include <assert.h>
int nondet_int(void);
int main(void)
{
  int m = nondet_int();
  __ESBMC_assume(m >= 2 && m <= 4);
  int a[3][m];
  a[1][1] = 7;
  a[2][m - 1] = 9;
  int (*p)[m] = a;
  int (**pp)[m] = &p;
  assert((*pp)[1][1] == 7);
  *pp = a + 2;
  assert((**pp)[m - 1] == 8);
  return 0;
}
