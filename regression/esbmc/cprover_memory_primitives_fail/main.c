#include <stdlib.h>

int main(void)
{
  char *p = malloc(8);
  if (!p)
    return 0;
  __ESBMC_assert(__CPROVER_OBJECT_SIZE(p) == 16, "wrong size on purpose");
  free(p);
  return 0;
}
