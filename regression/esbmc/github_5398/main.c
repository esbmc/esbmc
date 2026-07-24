#include <stdlib.h>

extern void __VERIFIER_assume(int);

// Regression for issue #5398: under --malloc-zero-is-null, malloc() with a
// compile-time-constant size of 0 must actually assign NULL to its lvalue.
// Previously the constant-0 branch returned NULL without assigning the lhs, so
// the pointer kept its uninitialised (invalid) value; the != NULL assume then
// wrongly passed and free() reported a spurious "invalid pointer freed".
int main(void)
{
  void *p = malloc(0);
  // With --malloc-zero-is-null, p == NULL, so this assumption prunes the path
  // and free() below is never reached: the program is memory-safe.
  __VERIFIER_assume((unsigned long)p != (unsigned long)0);
  free(p);
  return 0;
}
