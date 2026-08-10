#include <assert.h>
#include <stdlib.h>

void __CPROVER_havoc_object(void *);

int main(void)
{
  int *p = malloc(3 * sizeof(int));
  p[0] = 1;
  // A pointer value, not an address-of: the object is not statically known,
  // so the adapter declines instead of havocking the pointer itself.
  __CPROVER_havoc_object(p);
  assert(p[0] == 1);
  free(p);
  return 0;
}
