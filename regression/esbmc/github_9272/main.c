#include <stdint.h>

/* Casting a local's address back from an integer must yield the local, even
 * when the slicer drops every step taking its address before the cast. */
uintptr_t nondet_uintptr(void);
int main(void)
{
  int a;
  uintptr_t ua = (uintptr_t)&a;
  uintptr_t x = nondet_uintptr();
  int *p = (int *)x;
  if (x == ua)
    __ESBMC_assert(p == &a, "cast of a's address is &a");
  return 0;
}
