#include <assert.h>
int nondet_int(void);
int main(void)
{
  int m = nondet_int();
  __ESBMC_assume(m >= 2 && m <= 4);
  int a[2][m][3];
  int i = nondet_int(), j = nondet_int(), k = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2 && j >= 0 && j < m && k >= 0 && k < 3);
  a[i][j][k] = 7;
  a[1 - i][j][k] = 9;
  assert(a[i][j][k] == 7 && a[1 - i][j][k] == 9);
  return 0;
}
