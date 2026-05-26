#include <stdlib.h>

int main(void)
{
  int *p = calloc(1, sizeof(int));
  if (p == 0)
    return 1;
  *p = 42;
  free(p);
  return 0;
}
