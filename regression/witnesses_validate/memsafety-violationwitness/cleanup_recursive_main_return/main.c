#include <stdlib.h>
static int depth;
int main(void)
{
  if (depth++ == 0)
  {
    void *p = malloc(4);
    __ESBMC_assume(p != 0);
    main();
    return 0;
  }
  return 0;
}
