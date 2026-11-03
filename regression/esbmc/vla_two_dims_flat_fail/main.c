#include <assert.h>
int nondet_int(void);
long nondet_long(void);
int main(void)
{
  int m = nondet_int();
  long n = nondet_long();
  __ESBMC_assume(m >= 1 && m <= 3 && n >= 2 && n <= 3);
  int a[2][m][n][3];
  int i = nondet_int(), k = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2 && k >= 0 && k < 3);
  a[i][m - 1][1][k] = 5;
  a[i][m - 1][0][k] = 6;
  int *flat = &a[0][0][0][0];
  assert(flat[((i * m + m - 1) * n + 1) * 3 + k] == 6);
  return 0;
}
