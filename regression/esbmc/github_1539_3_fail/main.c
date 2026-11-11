#include <stdint.h>

/* The integer cast can land on a global whose address first appears in the
 * formula after the cast. */
int a;
uintptr_t nondet_uintptr(void);
int main(void)
{
  uintptr_t ua = (uintptr_t)&a;
  int *p = (int *)nondet_uintptr();
  __ESBMC_assert(p != &a || ua == 0, "p may point to a");
  return 0;
}
