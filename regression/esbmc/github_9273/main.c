#include <stdint.h>

/* #9273: a cast that lands on a still names a's address. */
int a;
uintptr_t nondet_uintptr(void);
_Bool nondet_bool(void);
int main(void)
{
  uintptr_t n = nondet_uintptr();
  int *p = (int *)n;
  uintptr_t ua = 1;
  if (nondet_bool())
    ua = (uintptr_t)&a;
  __ESBMC_assert(p != &a || n == (uintptr_t)&a, "p is a only at a's address");
  return 0;
}
