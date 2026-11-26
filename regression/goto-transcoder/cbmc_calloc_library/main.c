/* Built with goto-instrument --add-library, so calloc is CBMC's own model,
 * which allocates with __CPROVER_allocate(n, 1). */
#include <stdlib.h>
int main(void)
{
  int *p = calloc(4, sizeof(int));
  if (p)
    __CPROVER_assert(p[3] == 0, "calloc zero-fills");
  return 0;
}
