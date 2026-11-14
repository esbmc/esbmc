/* github #10303: printf and fprintf did not check that the format pointer
 * is valid. */
#include <stdio.h>
#include <stdlib.h>

int main()
{
  char *p = malloc(3);
  if (!p)
    return 0;
  p[0] = 'h';
  p[1] = '\n';
  p[2] = '\0';
  printf(p);
  fprintf(stdout, p);
  free(p);
  return 0;
}
