#include <string.h>
int nondet_int(void);

int main(void)
{
  int a[2][2][2];
  int b[2];
  int i = nondet_int();
  int j = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  __ESBMC_assume(j >= 0 && j < 2);

  a[0][0][0] = 1;
  a[0][0][1] = 1;
  a[0][1][0] = 1;
  a[0][1][1] = 1;
  a[1][0][0] = 1;
  a[1][0][1] = 1;
  a[1][1][0] = 1;
  a[1][1][1] = 1;
  b[0] = 5;
  b[1] = 6;

  memcpy(a[1][1], b, sizeof b);

  __ESBMC_assert(
    a[i][1][j] == (i == 1 ? (j == 0 ? 5 : 6) : 1),
    "the innermost grafted row reads back at both subscripts");
  return 0;
}
