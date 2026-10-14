/* Boundary for #5393: the same nondet pointer survives when the allocation
   carries a type, because the store is then a whole-element write rather than
   a byte-level one. malloc(sizeof(void *)) types the object; calloc, and any
   malloc whose size is a runtime value, do not. */
#include <stdlib.h>

void *nondet_voidp(void);

int main(void)
{
  void *q = nondet_voidp();
  __ESBMC_assume(q != 0);

  void **s = malloc(sizeof(void *));
  if (s == 0)
    return 0;

  *s = q;

  __ESBMC_assert(*s != 0, "a typed allocation keeps the pointer");
  return 0;
}
