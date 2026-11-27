#include <stdint.h>

/* Only sliced steps take a's and b's addresses, so their non-overlap is
 * asserted lazily, once a model overlaps them (#10796). The casts name both,
 * which the layout keeps apart. */
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
    __ESBMC_assert(sa + 3 < sb || sb + 3 < sa, "a and b do not overlap");
  }
  return 0;
}
