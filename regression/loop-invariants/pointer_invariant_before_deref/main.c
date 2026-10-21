#include <stdlib.h>
static unsigned long j;
int main() {
  unsigned n = nondet_uint(); __ESBMC_assume(n >= 1 && n <= 1000);
  char *src = malloc(n), *dst = malloc(n);
  const char *s = src; char *d = dst;
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= n && s == __ESBMC_loop_entry(s) + i && d == __ESBMC_loop_entry(d) + i &&
    __ESBMC_forall(&j, !(j < i) || (d - i)[j] == (s - i)[j]));
  while (i < n) { *d = *s; d++; s++; i++; }
}
