#include <stdlib.h>

int main(void)
{
  int *p = calloc(1, sizeof(int));
  *p = 42;
  free(p);
  return 0;
}
