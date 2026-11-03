#include <stdlib.h>
int main() {
  int *a = malloc(sizeof(int));
  if (!a) return 0;
  int *t = realloc(a, 2 * sizeof(int));
  if (t) a = t;
  free(a);
  return 0;
}
