#include <assert.h>
#include <stdio.h>

int main()
{
  // %o of 8 = "10" (2 chars)
  int r1 = printf("%o", (unsigned)8);
  assert(r1 == 2);

  // %o of 0 = "0" (1 char)
  int r2 = printf("%o", (unsigned)0);
  assert(r2 == 1);

  // %04o of 7 = "0007" (4 chars, zero-padded)
  int r3 = printf("%04o", (unsigned)7);
  assert(r3 == 4);
}
