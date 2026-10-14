#include <stdlib.h>
int main() {
  int *a = malloc(sizeof(int));
  if (!a) return 0;
  size_t n = nondet_ulong();
  int *t = realloc(a, n);
  if (t) a = t;
  free(a);
  return 0;
}
