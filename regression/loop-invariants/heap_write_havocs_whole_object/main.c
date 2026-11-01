#include <stdlib.h>
int main() {
  unsigned n = nondet_uint(); __ESBMC_assume(n >= 2 && n <= 1000);
  int *a = malloc(n * sizeof(int));
  a[1] = 5;
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= n);
  while (i < n) { a[i] = 7; i++; }
  __ESBMC_assert(a[1] == 5, "should fail: a[1] was overwritten");
}
