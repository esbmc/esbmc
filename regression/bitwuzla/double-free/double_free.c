#include <stdlib.h>
#include <stdio.h>

int main()
{
  int *i;
  i = (int *)malloc(sizeof(*i));
  printf("i=%p\n", i);
  free(i);
  printf("i=%p\n", i);
  free(i);
  printf("i=%p\n", i);

  return 0;
}
