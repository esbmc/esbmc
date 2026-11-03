#include <stdlib.h>
int main() {
  int *a = malloc(sizeof(int));
  if (!a) return 0;
  a = realloc(a, 2 * sizeof(int));
  free(a);
  return 0;
}
