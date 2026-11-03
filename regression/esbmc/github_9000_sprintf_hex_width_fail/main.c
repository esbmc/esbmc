/* #9000: sprintf measures a field width and an octal or hex conversion. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
unsigned short nondet_ushort();
int main()
{
  char d[8];
  unsigned short c = nondet_ushort();
  int r = sprintf(d, "%-2x", c);
  assert(r < 4);
  return 0;
}
