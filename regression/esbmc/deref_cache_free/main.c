#include <stdlib.h>

// The resolved dereference carries this access's validity claims. free()
// changes which objects are valid without touching the value set of p, so a
// memoised resolution must not survive it -- otherwise the use-after-free
// below is never checked.
int main()
{
  int *p = malloc(sizeof(int));
  *p = 42;
  free(p);
  *p = 99;
  return 0;
}
