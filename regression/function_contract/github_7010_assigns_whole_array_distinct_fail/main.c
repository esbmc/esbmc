// The whole-array havoc has to leave the elements independent. ARRAY_OF(NONDET)
// ties them to one value, so an ensures naming two different values had no
// post-state to land in and the path died before reaching the assert.
#include <assert.h>

int a[4];

void f(void)
{
  __ESBMC_assigns(a);
  __ESBMC_ensures(a[0] == 1 && a[1] == 2);
  a[0] = 1;
  a[1] = 2;
}

int main(void)
{
  f();
  assert(a[0] == 1 && a[1] == 2); /* holds under the contract */
  assert(0);                      /* reachable */
  return 0;
}
