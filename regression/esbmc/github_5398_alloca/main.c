// --malloc-zero-is-null is a statement about malloc: alloca has no NULL
// alternative to explore, so a zero-sized frame allocation is unaffected.
#include <stdlib.h>

int main(void)
{
  void *p = __builtin_alloca(0);
  __ESBMC_assert(p != 0, "alloca(0) is not NULL");
  return 0;
}
