/* #9000: sprintf measures a field width and an octal or hex conversion. */
#include <stdio.h>
#include <stdlib.h>
unsigned char nondet_uchar();
int main()
{
  char *d = malloc(5);
  unsigned char c = nondet_uchar();
  sprintf(d, "%5d", c % 10);
  free(d);
  return 0;
}
