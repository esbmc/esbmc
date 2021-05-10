#include <stdlib.h>
#include <stdio.h>

#define SIZE 13

void main(void)
{
  union
  {
    int *p0;
    struct
    {
      char p1[SIZE];
      int p2;
    } str;
  } data;
  data.p0 = (int *)malloc(sizeof(int *));
  data.str.p2 = 1;
  free(data.p0);
}
