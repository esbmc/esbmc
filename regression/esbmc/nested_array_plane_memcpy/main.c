#include <string.h>
int nondet_int(void);

int main(void)
{
  int a[2][2][2];
  int b[2][2];
  int i = nondet_int();
  int j = nondet_int();
  __ESBMC_assume(i >= 0 && i < 2);
  __ESBMC_assume(j >= 0 && j < 2);

  a[1][0][0] = 1;
  a[1][0][1] = 1;
  a[1][1][0] = 1;
  a[1][1][1] = 1;
  b[0][0] = 5;
  b[0][1] = 6;
  b[1][0] = 7;
  b[1][1] = 8;

  memcpy((char *)a + sizeof a[0], b, sizeof a[0]);

  __ESBMC_assert(
    a[1][i][j] >= 5 && a[1][i][j] <= 8, "the grafted plane reads back");
  return 0;
}
