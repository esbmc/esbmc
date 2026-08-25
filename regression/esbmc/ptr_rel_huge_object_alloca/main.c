/* R38: the PTRDIFF_MAX cap applies to malloc only, so alloca can still lay out
   an object whose upper offsets read negative in the pointer comparator. The
   malloc spelling of this program verifies (ptr_rel_huge_object); this one
   reports a spurious counterexample at n = 0x8000000000000000. The width is
   fixed: `unsigned long` is 32-bit on LLP64 and never reaches 2^63. */
#include <assert.h>
#include <alloca.h>
#include <stdint.h>

uint64_t nondet_uint64(void);

int main(void)
{
  uint64_t n = nondet_uint64();

  char *p = alloca(n);
  if (!p)
    return 0;

  char *q = p + n;

  assert(q >= p);
  return 0;
}
