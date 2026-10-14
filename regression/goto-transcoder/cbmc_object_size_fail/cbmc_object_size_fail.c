#include <stdlib.h>

int main()
{
  char *p = malloc(16);
  if (!p) return 0;
  __CPROVER_assert(__CPROVER_OBJECT_SIZE(p) == 15, "object size is 15");
  return 0;
}
