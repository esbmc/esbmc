#include <stdlib.h>

union
{
  int *p0;
  int p1;
} data;

void main(void)
{
  data.p0 = (int *)malloc(sizeof(int));
  data.p1 = 1;
  void *ptr = data.p0;
  free(ptr);
}
