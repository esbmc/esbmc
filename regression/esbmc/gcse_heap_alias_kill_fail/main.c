#include <assert.h>
#include <stdlib.h>

// p and q point to the same heap object, so the store through p must kill
// `*q + 1` (#7992).
int main()
{
  int *p = malloc(sizeof(int));
  __ESBMC_assume(p != 0);
  int *q = p;
  *p = 1;
  int y = *q + 1;
  *p = 2;
  int z = *q + 1;
  assert(z == y);
  free(p);
}
