#include <stdint.h>

/* As github_10796, but a and b may be laid out next to each other, so the
 * lazily asserted layout must still leave that counterexample (#10796). */
uintptr_t nondet_uintptr(void);
int main(void)
{
  char a[3], b[3];
  char *pa = a, *pb = b;
  (void)pa;
  (void)pb;
  uintptr_t x = nondet_uintptr(), y = nondet_uintptr();
  char *p = (char *)x, *q = (char *)y;
  if (__ESBMC_same_object(p, a) && __ESBMC_same_object(q, b))
  {
    uintptr_t sa = x - __ESBMC_POINTER_OFFSET(p);
    uintptr_t sb = y - __ESBMC_POINTER_OFFSET(q);
    __ESBMC_assert(sa + 4 < sb || sb + 4 < sa, "a and b are not adjacent");
  }
  return 0;
}
