// Anti-vacuity twin: the offset of a[1][2][2] is 3*m + 8, so an equality
// against 3*m + 9 elements past a[0][0][0] has to be refuted.
#include <assert.h>

int main(void)
{
  int m = nondet_int();
  __ESBMC_assume(m >= 3 && m <= 5);

  int a[2][m][3];
  int *p = &a[1][2][2];
  int *q = &a[0][0][0] + (3 * m + 9);

  assert(p == q);
  return 0;
}
