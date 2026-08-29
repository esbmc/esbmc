// The NULL alternative of malloc(0) must stay a genuine NULL pointer: the
// allocation tracking arrays are indexed by the dynamic object, never by the
// conditional result, or __ESBMC_alloc[NULL] is set and this deref is missed.
#include <stdlib.h>

int main(void)
{
  char *p = malloc(0);
  if (p == 0)
    *p = 1;
  return 0;
}
