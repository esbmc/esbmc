// #10303: printf reads a format string whose heap object was freed.
#include <stdio.h>
#include <stdlib.h>

int main()
{
  char *p = (char *)malloc(sizeof(char));
  free(p);
  printf(p, 0);
}
