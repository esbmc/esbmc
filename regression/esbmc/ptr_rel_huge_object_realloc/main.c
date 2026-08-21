/* R38, the realloc half: symex_realloc never bounds its size at all, so the
   same offset at 2^63 reaches the comparator. See ptr_rel_huge_object for the
   malloc spelling, which the PTRDIFF_MAX cap fixed. */
#include <assert.h>
#include <stdlib.h>

unsigned long nondet_ulong(void);

int main(void)
{
  unsigned long n = nondet_ulong();

  char *p = malloc(1);
  if (!p)
    return 0;

  p = realloc(p, n);
  if (!p)
    return 0;

  char *q = p + n;

  assert(q >= p);
  return 0;
}
