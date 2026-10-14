#include <stdlib.h>

int main()
{
  char *p = malloc(8);
  if (!p) return 0;
  __CPROVER_assert(__CPROVER_r_ok(p, 8), "8 bytes are readable");
  __CPROVER_assert(__CPROVER_w_ok(p, 8), "8 bytes are writable");
  return 0;
}
