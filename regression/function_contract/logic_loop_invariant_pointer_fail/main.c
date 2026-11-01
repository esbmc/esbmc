#include <stdlib.h>
__ESBMC_logic long X(const char *p, long r) { return r <= 0 ? 0 : 1 + X(p + 1, r - 1); }
int main() {
  long n = nondet_long();
  __ESBMC_assume(n >= 0 && n < 100);
  char *s = malloc(n);
  const char *p = s, *end = s + n;
  __ESBMC_loop_invariant(s <= p && p <= end && __ESBMC_same_object(p, end) && (p - s) + X(p, end - p) == X(s, n) + 1);
  for (; p < end; p++) ;
  return 0;
}
