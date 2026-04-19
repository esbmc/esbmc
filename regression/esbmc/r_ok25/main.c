#include <assert.h>
#include <stdlib.h>

int main()
{
  int *p = malloc(4);
  assert(__ESBMC_r_ok(p, 1));
  return 0;
}
