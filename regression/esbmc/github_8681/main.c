/* esbmc/esbmc#8681: a guarded index into the free operand passes its checks. */
#include <stdlib.h>
int nondet_int();
int main(void)
{
  int *a[2] = {0, malloc(sizeof(int))};
  int k = nondet_int();
  __ESBMC_assume(k == 1 || k == 2);
  free(a[2 / (k + 1)]);
  return 0;
}
