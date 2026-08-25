/* R38, the realloc half: symex_realloc never bounds its size at all, so the
   same offset at 2^63 reaches the comparator. See ptr_rel_huge_object for the
   malloc spelling, which the PTRDIFF_MAX cap fixed. The width is fixed:
   `unsigned long` is 32-bit on LLP64 and never reaches 2^63. */
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>

uint64_t nondet_uint64(void);

int main(void)
{
  uint64_t n = nondet_uint64();

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
