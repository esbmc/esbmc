/* Boundary for #5393: a pointer the value set can track survives the same
   untyped-byte-allocation round trip that loses a nondet one. */
#include <stdlib.h>

int main(void)
{
  int x;
  void *q = &x;

  char *raw = malloc(64);
  if (raw == 0)
    return 0;

  void **s = (void **)raw;
  *s = q;

  __ESBMC_assert(*s == q, "a concrete pointer round-trips through bytes");
  return 0;
}
