/* R37: pointer_struct's offset member is ptraddr_type2() -- full unsigned
   width -- so an offset at or above 2^63 reads negative under the signed
   comparison and one-past-the-end sorts below the base. Reaching it needs an
   8 EiB allocation, which is why R36 takes the signed reading anyway. The
   width is fixed: `unsigned long` is 32-bit on LLP64 and never reaches 2^63. */
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
