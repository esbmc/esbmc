#include <stdlib.h>
int main(void)
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n > 0 && n < 4);
  int (*p)[n] = malloc(4 * n * sizeof(int));
  if (!p)
    return 0;
  unsigned j = nondet_uint();
  __ESBMC_assume(j < 4);
  p[j][0] = 7;
  __ESBMC_assert(p[j][0] == 8, "vla row");
  free(p);
  return 0;
}
