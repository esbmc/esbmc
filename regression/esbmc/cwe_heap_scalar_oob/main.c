#include <stdlib.h>

int main(void)
{
  int *p = malloc(sizeof(int));
  if (!p)
    return 0;
  p[3] = 42; /* OOB on a size-1 heap object -> Access to object out of bounds */
  free(p);
  return 0;
}
