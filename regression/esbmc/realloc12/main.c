#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

int main() {
  int size = __VERIFIER_nondet_int();
  if (size <= 0 || size > 50) return 0;

  int *p = NULL;
  p = realloc(p, size * sizeof(int)); // should behave like malloc
  if (p == NULL) return 0;            // allocation may fail

  p[0] = 42;                          // must be safe if allocation succeeded
  free(p);
  return 0;
}

