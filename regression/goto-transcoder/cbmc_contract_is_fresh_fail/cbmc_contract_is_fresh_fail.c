#include <stdlib.h>

int read_second(int *p)
  __CPROVER_requires(__CPROVER_is_fresh(p, sizeof(int)))
{
  return p[1];
}

int main()
{
  int *q = malloc(sizeof(int));
  if (!q) return 0;
  return read_second(q);
}
