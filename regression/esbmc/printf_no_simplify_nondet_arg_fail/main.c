#include <assert.h>
#include <stdio.h>

unsigned nondet_uint(void);

int main()
{
  /* Over-pinning control: %02X of a genuinely unknown value spans 2..8
     characters, so the return value must stay a range, not fold to a
     constant. */
  unsigned v = nondet_uint();
  int r = printf("%02X", v);
  assert(r == 2);
}
