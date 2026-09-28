#include <stdlib.h>

int *alloc_one(void)
  __CPROVER_ensures(__CPROVER_is_fresh(__CPROVER_return_value, sizeof(int)))
{
  return malloc(sizeof(int));
}

int main()
{
  int *p = alloc_one();
  p[1] = 5;
  __CPROVER_assert(p[1] == 5, "caller can write the promised fresh object");
  return 0;
}
