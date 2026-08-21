/* R37: pointer_struct's offset member is ptraddr_type2() -- full unsigned
   width -- so an offset at or above 2^63 reads negative under the signed
   comparison and one-past-the-end sorts below the base. Reaching it needs an
   8 EiB allocation, which is why R36 takes the signed reading anyway. */
#include <assert.h>
#include <stdlib.h>

unsigned long nondet_ulong(void);

int main(void)
{
  unsigned long n = nondet_ulong();

  char *p = malloc(n);
  if (!p)
    return 0;

  char *q = p + n;

  assert(q >= p);
  return 0;
}
