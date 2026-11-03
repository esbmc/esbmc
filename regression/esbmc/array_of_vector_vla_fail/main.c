#include <assert.h>
typedef int v4i __attribute__((__vector_size__(16)));
extern unsigned nondet_uint(void);
int main(void)
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n > 1 && n < 4);
  v4i g[n][2];
  unsigned i = nondet_uint(), j = nondet_uint(), m = nondet_uint();
  __ESBMC_assume(i < n && j < 2 && m < 4);
  g[i][j] = (v4i){0, 0, 0, 0};
  g[i][j][m] = 6;
  assert(g[i][j][(m + 1) % 4] == 6);
  return 0;
}
