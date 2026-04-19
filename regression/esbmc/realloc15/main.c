#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

int main() {
  int n = __VERIFIER_nondet_int();
  if (n < 1 || n > 5) return 0;

  int *p = malloc(n * sizeof(int));
  if (!p) return 0;

  for (int i = 0; i < n; i++) p[i] = i + 10;

  int newn = __VERIFIER_nondet_int();
  if (newn <= n || newn > 12) return 0; // grow only

  p = realloc(p, newn * sizeof(int));
  if (!p) return 0;

  for (int i = 0; i < n; i++) {
    __ESBMC_assert(p[i] == i + 10, "Grow must preserve old data");
  }
  // p[n..newn-1] are uninitialized: we do not assert them

  free(p);
  return 0;
}

