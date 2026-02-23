#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

int main() {
  int n = __VERIFIER_nondet_int();
  if (n < 3 || n > 10) return 0;

  int *p = malloc(n * sizeof(int));
  if (!p) return 0;

  for (int i = 0; i < n; i++) p[i] = i + 1;

  int newn = __VERIFIER_nondet_int();
  if (newn <= 0 || newn >= n) return 0; // shrink only

  p = realloc(p, newn * sizeof(int));
  if (!p) return 0;

  for (int i = 0; i < newn; i++) {
    __ESBMC_assert(p[i] == i + 1, "Shrink must preserve old data");
  }

  free(p);
  return 0;
}

