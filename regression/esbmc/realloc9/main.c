#include <stdlib.h>

int main() {
  int *p = malloc(5 * sizeof(int));
  if (!p) return 0;
  for (int i = 0; i < 5; i++) p[i] = i + 1;
  p = realloc(p, 3 * sizeof(int)); // shrink
  if (!p) return 0;
  // First 3 elements must be preserved
  if (p[0] != 1 || p[1] != 2 || p[2] != 3) {
    __ESBMC_assert(0, "Shrink failed to preserve data");
  }
  free(p);
  return 0;
}

