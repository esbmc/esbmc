#include <assert.h>
typedef int v4i __attribute__((__vector_size__(16)));
extern unsigned nondet_uint(void);
extern int nondet_int(void);
int main(void)
{
  v4i g[2][3] = {};
  unsigned i = nondet_uint(), j = nondet_uint(), m = nondet_uint();
  unsigned i2 = nondet_uint(), j2 = nondet_uint(), m2 = nondet_uint();
  int n = nondet_int();
  __ESBMC_assume(i < 2 && j < 3 && m < 4 && n != 0);
  __ESBMC_assume(i2 < 2 && j2 < 3 && m2 < 4);
  g[i][j][m] = n;
  assert(g[i][j][m] == n && ((i == i2 && j == j2 && m == m2) || g[i2][j2][m2] == 0));
  return 0;
}
