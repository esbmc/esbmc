/* #8048: an object whose size fits keeps its exact layout, so it cannot wrap
 * around the end of the address space. */
#include <assert.h>
#include <stdint.h>
int __VERIFIER_nondet_int(void);
int main()
{
  int n = __VERIFIER_nondet_int();
  __ESBMC_assume(n > 1 && n < 8);
  int a[n];
  assert((uintptr_t)&a[n - 1] > (uintptr_t)a);
  return 0;
}
