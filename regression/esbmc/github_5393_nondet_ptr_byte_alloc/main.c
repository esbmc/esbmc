/* Storing a pointer into an untyped byte allocation takes it through its
   numeric address. Only the NULL object has address zero, but that invariant
   only covered objects symex had registered, so a pointer whose object id is
   a free variable could be placed at address 0 and the assumed non-nullness
   was lost on the read back (#7008).

   The three sibling tests pin the boundary: concrete pointers, integers and
   typed allocations survive the same shape, and did so before the fix. */
#include <stdlib.h>

void *nondet_voidp(void);

int main(void)
{
  void *q = nondet_voidp();
  __ESBMC_assume(q != 0);

  char *raw = malloc(64);
  if (raw == 0)
    return 0;

  void **s = (void **)raw;
  *s = q;

  __ESBMC_assert(*s != 0, "a stored non-null pointer reads back non-null");
  return 0;
}
