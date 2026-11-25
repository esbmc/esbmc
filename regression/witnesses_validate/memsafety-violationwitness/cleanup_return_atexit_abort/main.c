#include <stdlib.h>
static void *p;
static void cleanup(void) { abort(); }
int main(void)
{
  p = malloc(4);
  __ESBMC_assume(p != 0);
  atexit(cleanup);
  return 0;
}
