#include <stdlib.h>

int main()
{
 
  int *foobar = malloc(12);
  __ESBMC_assume(foobar);
  foobar[0] = 0;
  foobar[2] = 1;
  assert(foobar[0] == 0);
  return 0;
}
