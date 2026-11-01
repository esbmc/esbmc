/* #9354: fputs_unlocked reads a freed string. */
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
  char *p = malloc(4);
  if (!p)
    return 0;
  p[0] = 'a';
  p[1] = 0;
  free(p);
  fputs_unlocked(p, stdout);
  return 0;
}
