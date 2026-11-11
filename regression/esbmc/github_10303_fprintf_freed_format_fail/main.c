// #10303: fprintf reads a format string whose heap object was freed.
#include <stdio.h>
#include <stdlib.h>

int main()
{
  char *p = (char *)malloc(4);
  if (!p)
    return 0;
  p[0] = '%';
  p[1] = 'd';
  p[2] = '\n';
  p[3] = '\0';
  free(p);
  fprintf(stdout, p, 1);
}
