#include <stdlib.h>

int main(void)
{
  int *p = malloc(4 * sizeof(int));
  if (!p)
    return 0;
  p[5] = 42; /* out-of-bounds write on a heap buffer (index 5, size 4) */
  free(p);
  return 0;
}
