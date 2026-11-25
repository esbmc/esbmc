/* #9221: object layouts are asserted at the first pointer/integer cast, each
 * under the __ESBMC_alloc version its object was registered with. p was alive
 * when q was allocated, so they stay apart after p is freed. */
#include <stdint.h>
#include <stdlib.h>
int main(void)
{
  int *p = malloc(sizeof(int));
  int *q = malloc(sizeof(int));
  if (!p || !q)
    return 0;
  free(p);
  __ESBMC_assert((uintptr_t)p != (uintptr_t)q, "p and q were both alive");
  free(q);
  return 0;
}
