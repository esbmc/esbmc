/* #9000: sprintf measures a field width and an octal or hex conversion. */
#include <stdio.h>
#include <stdlib.h>
unsigned char nondet_uchar();
int main()
{
  char *d = malloc(4);
  unsigned char c = nondet_uchar();
  int r = sprintf(d, "\\%03o", c);
  free(d);
  return 0;
}
