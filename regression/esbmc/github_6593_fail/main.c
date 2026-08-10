#include <stdlib.h>

int main(void)
{
  void *p = malloc(0);
  free(p);

  /* malloc(0) is NULL under this flag, so asserting otherwise must be
   * reported -- if p were unconstrained again this would silently pass. */
  __ESBMC_assert(p != 0, "malloc(0) is not NULL");
  return 0;
}
