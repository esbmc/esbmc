#include <stdlib.h>
int main()
{
  int *p = malloc(sizeof(int));
  if (!p)
    return 0;
  *p = 5;
  free(p);
  return 0;
}
