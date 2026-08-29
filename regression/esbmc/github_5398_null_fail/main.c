#include <stdlib.h>

void reach_error(void)
{
  __ESBMC_assert(0, "NULL alternative of malloc(0) reached");
}

int main(void)
{
  void *p = malloc(0);
  if ((unsigned long)p == (unsigned long)0)
    reach_error();
  return 0;
}
