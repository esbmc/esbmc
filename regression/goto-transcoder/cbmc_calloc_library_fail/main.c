#include <stdlib.h>
int main(void)
{
  int *p = calloc(4, sizeof(int));
  if (p)
    __CPROVER_assert(p[3] == 1, "calloc'd memory does not hold 1");
  return 0;
}
