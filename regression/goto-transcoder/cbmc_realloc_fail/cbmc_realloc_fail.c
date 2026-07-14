#include <stdlib.h>
int main()
{
  int *p = malloc(2 * sizeof(int));
  if (!p)
    return 0;
  int *q = (int *)realloc(p, 4 * sizeof(int));
  if (!q)
    return 0;
  q[8] = 9; // out of bounds
  return 0;
}
