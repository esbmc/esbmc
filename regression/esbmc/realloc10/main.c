#include <stdlib.h>

int main() {
  int *p = malloc(2 * sizeof(int));
  if (!p) return 0;
  p[0] = 10;
  p[1] = 20;
  p = realloc(p, 4 * sizeof(int)); // grow
  if (!p) return 0;
  __ESBMC_assert(p[0] == 10 && p[1] == 20, "Grow must preserve old data");
  // p[2], p[3] are uninitialized -> should not be read
  free(p);
  return 0;
}

