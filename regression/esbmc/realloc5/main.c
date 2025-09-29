#include <stdlib.h>

int main() {
  int *p = malloc(5 * sizeof(int));
  if (!p) return 0;
  p[0] = 1;
  int *q = realloc(p, 0); // should free p and return NULL
  if (q != NULL) return 1; // violation if realloc(,0) != NULL
  return 0;
}

