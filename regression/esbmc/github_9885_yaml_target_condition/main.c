#include <stdlib.h>

int main(void)
{
  char *p = malloc(10);
  if (!p)
    return 0;
  free(p);
  if (p[2] == 'a')
    return 1;
  return 0;
}
