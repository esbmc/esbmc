/* The byte-reversed value the scatter used to store must not be readable. */
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  int *v = (int *)malloc(sizeof(int) * 4);
  __ESBMC_assume(v);
  v[0] = 0x01020304;
  assert(v[0] == 0x04030201);
  return 0;
}
