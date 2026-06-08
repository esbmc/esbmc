#include <assert.h>
#include <stdio.h>

int main()
{
  // %x of 0xab = "ab" (2 chars)
  int r1 = printf("%x", (unsigned)0xab);
  assert(r1 == 2);

  // %X of 0xAB = "AB" (2 chars)
  int r2 = printf("%X", (unsigned)0xAB);
  assert(r2 == 2);

  // %04x of 0xa = "000a" (4 chars, zero-padded)
  int r3 = printf("%04x", (unsigned)0xa);
  assert(r3 == 4);

  // %6X of 0xAB = "    AB" (6 chars, space-padded)
  int r4 = printf("%6X", (unsigned)0xAB);
  assert(r4 == 6);
}
