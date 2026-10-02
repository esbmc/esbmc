/* SV-COMP exempts only malloc and alloca from failing, so calloc may return
 * NULL under --sv-comp; --force-malloc-success alone still forces it. */
#include <assert.h>
#include <stdlib.h>
int main()
{
  int *p = calloc(1, sizeof(int));
  assert(p != NULL);
  free(p);
  return 0;
}
