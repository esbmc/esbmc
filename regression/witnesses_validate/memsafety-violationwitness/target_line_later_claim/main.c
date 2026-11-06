#include <stdlib.h>

int g(void)
{
  return 1;
}

int main()
{
  char *p = malloc(10);
  char *q = malloc(10);
  if (!p || !q)
    return 0;
  free(p);
  if (q[0] == 'a' && g() && p[2] == 'a')
    return 1;
  return 0;
}
