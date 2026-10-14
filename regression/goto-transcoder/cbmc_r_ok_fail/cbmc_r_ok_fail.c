#include <stdlib.h>

int main()
{
  char *p = malloc(8);
  if (!p) return 0;
  __CPROVER_assert(__CPROVER_r_ok(p, 16), "16 bytes are readable");
  return 0;
}
