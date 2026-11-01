#include <stdlib.h>
static unsigned long j;
int main() {
  unsigned n = nondet_uint(); __ESBMC_assume(n >= 1 && n <= 1000);
  int *a = malloc(n * sizeof(int));
  int a0 = a[0]; __ESBMC_assume(a0 < 100);
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= n &&
    __ESBMC_forall(&j, !(j < i) || a[j] == __ESBMC_loop_entry(a[j]) + 1) &&
    __ESBMC_forall(&j, !(j >= i && j < n) || a[j] == __ESBMC_loop_entry(a[j])));
  while (i < n) { a[i] = a[i] + 1; i++; }
  __ESBMC_assert(a[0] == a0 + 1, "bumped");
}
