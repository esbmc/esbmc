#include <stdlib.h>

extern void __VERIFIER_assume(int);

// Issue #5398: under --malloc-zero-is-null, a compile-time-constant malloc(0)
// must assign NULL to its lvalue, or the assume below wrongly passes.
int main(void)
{
  void *p = malloc(0);
  // p == NULL, so this prunes the path and free() is never reached.
  __VERIFIER_assume((unsigned long)p != (unsigned long)0);
  free(p);
  return 0;
}
