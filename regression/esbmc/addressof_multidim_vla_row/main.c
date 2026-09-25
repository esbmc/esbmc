// The middle dimension is a VLA, so the enclosing row's array size is not a
// constant and no stride can be computed; the walk has to stop there rather
// than read the size as a constant (#6778). Comparing against the offset
// spelled out symbolically is what makes a wrong stride observable — a
// disequality alone would hold for any non-zero offset, right or wrong.
#include <assert.h>

int main(void)
{
  int m = nondet_int();
  __ESBMC_assume(m >= 3 && m <= 5);

  int a[2][m][3];
  int *p = &a[1][2][2];
  int *q = &a[0][0][0] + (3 * m + 8);

  assert(p == q);
  return 0;
}
