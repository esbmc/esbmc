#include <assert.h>
int a[2] = {1, 2};
int b[2];
int main()
{
  int *p = nondet_bool() ? a : b;
  __ESBMC_assume(p == b);
  __ESBMC_init_object(p);
  assert(a[0] == 1);
  return 0;
}
