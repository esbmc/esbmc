#include <stdlib.h>

int main()
{
  char *p = malloc(16);
  if (!p) return 0;
  __CPROVER_assert(__CPROVER_OBJECT_SIZE(p) == 16, "object size is 16");
  return 0;
}
