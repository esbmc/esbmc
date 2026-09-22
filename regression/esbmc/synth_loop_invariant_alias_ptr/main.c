/* P1: entry value of the accumulator is overwritten through a pointer between
   the constant assignment and the loop head. entry_value only compares
   assign.target against the symbol, so `*p = 5` is skipped and it reports 0. */
#include <stdint.h>
#include <assert.h>
int main(void) {
  uint32_t n; uint64_t a;
  __ESBMC_assume(n >= 1 && n <= 3);
  __ESBMC_assume(a <= 3);
  uint64_t s = 0;
  uint64_t *p = &s;
  *p = 5;
  uint64_t i = 1;
  while (i <= n) { s = s + a; i++; }
  assert(s == 5 + (uint64_t)n * a);
  return 0;
}
