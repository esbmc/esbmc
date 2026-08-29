#include <assert.h>
#include <stdlib.h>

int main()
{
  int *p = malloc(4 * sizeof(int));
  if (p == NULL)
    return 0;

  /* Over-classification control: a non-zero size must not take the
     free-and-return-NULL path. */
  int *q = realloc(p, 8 * sizeof(int));
  assert(q == NULL);
}
