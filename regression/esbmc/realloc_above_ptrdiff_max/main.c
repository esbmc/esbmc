/* An over-cap realloc must fail, and C17 7.22.3.5 requires the old object to
   be unchanged when it does. The cap therefore joins realloc's failure
   condition: nulling the result afterwards would leave the old object
   invalidated on the very branch that reports failure. */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

int main(void)
{
  char *p = malloc(4);
  if (!p)
    return 0;
  p[0] = 7;

  char *q = realloc(p, (size_t)PTRDIFF_MAX + 1);
  assert(q == NULL);

  assert(p[0] == 7);
  free(p);
  return 0;
}
