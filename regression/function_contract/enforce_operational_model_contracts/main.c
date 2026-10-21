#include <string.h>
#include <stdlib.h>
int main() {
  unsigned n = nondet_uint();
  __ESBMC_assume(n >= 1 && n <= 8);
  char *p = malloc(n);
  p[n - 1] = 0;
  size_t k = strlen(p);
  __ESBMC_assert(k < n && p[k] == 0, "len");
  free(p);
}
