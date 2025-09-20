#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

int main() {
  int size = __VERIFIER_nondet_int();
  if (size <= 0 || size > 20) return 0;

  int *p = malloc(size * sizeof(int));
  if (!p) return 0;

  p[0] = 1;
  int *q = realloc(p, 0);   // should free p and return NULL
  __ESBMC_assert(q == NULL, "realloc(ptr, 0) must return NULL");
  return 0;
}

