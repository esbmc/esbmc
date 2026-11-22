#include <assert.h>
#include <stdio.h>
int a[2] = {1, 2};
int b[2];
int main()
{
  int *p = nondet_bool() ? a : b;
  __ESBMC_assume(p == b);
  sscanf("5", "%d", p);
  assert(a[0] == 1);
  return 0;
}
