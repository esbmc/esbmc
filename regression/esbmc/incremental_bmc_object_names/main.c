#include <stdlib.h>
#include <assert.h>

/* --incremental-bmc runs one k iteration per bmct but shares the context, so
 * object counters reset per exploration would let a later iteration re-mint a
 * name an earlier one bound at a different type. The two then meet in one
 * formula as operands of different sort width. */

int main()
{
  unsigned n = 0;
  for (unsigned i = 0; i < 3; i++)
  {
    int *p = malloc(sizeof(int) * (i + 1));
    if (!p)
      return 0;
    p[0] = (int)i;
    n += (unsigned)p[0];
    free(p);
  }
  assert(n == 3);
  return 0;
}
