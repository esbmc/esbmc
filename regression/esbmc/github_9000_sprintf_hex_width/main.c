/* #9000: sprintf measures a field width and an octal or hex conversion. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
unsigned short nondet_ushort();
int main()
{
  char *d = malloc(5);
  unsigned short c = nondet_ushort();
  int r = sprintf(d, "%-2X", c);
  assert(r >= 2 && r <= 4);
  assert(r == 4 || c < 0x1000);
  free(d);
  return 0;
}
