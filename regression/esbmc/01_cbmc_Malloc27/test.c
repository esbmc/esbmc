#include <stdio.h>
#include <stdlib.h>

int main()
{
  int *p, *q;

  p = malloc(sizeof(int)*5);

  q = p+2;

  p[2] = 1;

  printf("*q: %d\n", *q);

  free(p);

  printf("*q: %d\n", *q);

  return 0;
}
