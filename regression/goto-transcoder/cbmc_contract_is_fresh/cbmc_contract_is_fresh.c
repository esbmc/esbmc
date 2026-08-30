#include <stdlib.h>

int read_first(int *p)
  __CPROVER_requires(__CPROVER_is_fresh(p, sizeof(int)))
  __CPROVER_ensures(__CPROVER_return_value == *p)
{
  return *p;
}

int main()
{
  int *q = malloc(sizeof(int));
  if (!q) return 0;
  *q = 42;
  return read_first(q);
}
