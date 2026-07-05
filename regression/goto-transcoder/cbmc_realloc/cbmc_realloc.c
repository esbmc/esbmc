#include <stdlib.h>
#include <assert.h>
int main()
{
  int *p = malloc(2 * sizeof(int));
  if (!p)
    return 0;
  p[0] = 1;
  p[1] = 2;
  int *q = (int *)realloc(p, 4 * sizeof(int));
  if (!q)
    return 0;
  assert(q[0] == 1); // data preserved across realloc
  q[3] = 9;          // valid: grown region
  return 0;
}
