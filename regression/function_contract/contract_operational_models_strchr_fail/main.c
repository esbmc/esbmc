#include <string.h>
#include <stdlib.h>
int main() {
  unsigned n = nondet_uint(); __ESBMC_assume(n >= 2 && n <= 6);
  char *p = malloc(n);
  p[n - 1] = 0; p[0] = 'x';
  char *e = strchr(p, '.');
  __ESBMC_assert(e == NULL || (*e == '.' && e > p), "dot");
  __ESBMC_assert(strchr(p, 'x') == p, "first");
  __ESBMC_assert(strchr(p, 0) != NULL, "nul");

  __ESBMC_assert(e == NULL, "bad");

  free(p);
}
