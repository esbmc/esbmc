#include <assert.h>
#include <stdlib.h>

int main()
{
  int *p = malloc(4 * sizeof(int));
  if (p == NULL)
    return 0;

  /* The zero-size classification must not depend on --no-simplify. */
  int *q = realloc(p, 0);
  assert(q == NULL);
}
