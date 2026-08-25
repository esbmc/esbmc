/* R38's residual: --force-malloc-success bounds a symbolic size only at
   max_layable_size(), so an object above PTRDIFF_MAX is still laid out and its
   one-past-the-end offset reads negative in the pointer comparator. The
   default path caps at PTRDIFF_MAX and verifies -- see ptr_rel_huge_object,
   which is this program without the flag. The width is fixed: `unsigned long`
   is 32-bit on LLP64 and never reaches 2^63. */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

uint64_t nondet_uint64(void);

int main(void)
{
  uint64_t n = nondet_uint64();

  char *p = malloc(n);
  if (!p)
    return 0;

  char *q = p + n;

  assert(q >= p);
  return 0;
}
