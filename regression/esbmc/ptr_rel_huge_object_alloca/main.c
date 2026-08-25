/* The alloca half of R38: an object above PTRDIFF_MAX would put upper offsets
   in the below-base window of the pointer comparator. alloca has no failure
   outcome, so the bound is assumed rather than branched on. The width is
   fixed: `unsigned long` is 32-bit on LLP64 and never reaches 2^63.
   Spelt __builtin_alloca because Windows ships no <alloca.h>. */
#include <assert.h>
#include <stdint.h>

uint64_t nondet_uint64(void);

int main(void)
{
  uint64_t n = nondet_uint64();

  char *p = __builtin_alloca(n);
  if (!p)
    return 0;

  char *q = p + n;

  assert(q >= p);
  return 0;
}
