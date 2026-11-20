#include <assert.h>
int a[2] = {1, 2};
int b[2];
int main()
{
  int *p = nondet_bool() ? a : b;
  __ESBMC_assume(p == b);
  __ESBMC_init_object(p);
  assert(b[0] == 0);
  return 0;
}
