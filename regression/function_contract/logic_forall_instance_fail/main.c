#include <stdlib.h>
__ESBMC_logic long ps(const int *s, long n) { return n <= 0 ? 0 : ps(s, n - 1) + s[n - 1]; }
int main() {
  long n = nondet_long();
  __ESBMC_assume(n >= 0 && n < 1000000);
  int *src = malloc(n * sizeof(int));
  long *out = malloc(n * sizeof(long));
  long acc = 0;
  long i = 0;
  long k;
  __ESBMC_loop_invariant(0 <= i && i <= n && acc == ps(src, i) &&
    __ESBMC_forall(&k, !(0 <= k && k < i) || out[k] == ps(src, k + 1)));
  for (; i < n; i++) { acc += src[i]; out[i] = acc; }
  long q;
  __ESBMC_assert(__ESBMC_forall(&q, !(0 <= q && q < n) || out[q] == ps(src, q)), "post");
  return 0;
}
