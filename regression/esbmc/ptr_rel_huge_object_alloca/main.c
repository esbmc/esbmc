/* R38: the PTRDIFF_MAX cap applies to malloc only, so alloca can still lay out
   an object whose upper offsets read negative in the pointer comparator. The
   malloc spelling of this program verifies (ptr_rel_huge_object); this one
   reports a spurious counterexample at n = 0x8000000000000000. */
#include <assert.h>
#include <alloca.h>

unsigned long nondet_ulong(void);

int main(void)
{
  unsigned long n = nondet_ulong();

  char *p = alloca(n);
  if (!p)
    return 0;

  char *q = p + n;

  assert(q >= p);
  return 0;
}
