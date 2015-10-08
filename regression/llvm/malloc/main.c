#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

float foo1() { return 10; }

int main()
{
  int *i = malloc(sizeof(int));
  int *i1 = malloc(sizeof(foo1()));

  return 0;
}
