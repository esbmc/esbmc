#include <stdlib.h>
int main()
{
  int *p = malloc(sizeof(int));
  if (!p)
    return 0;
  *p = 5;
  free(p);
  *p = 6; // use-after-free
  return 0;
}
