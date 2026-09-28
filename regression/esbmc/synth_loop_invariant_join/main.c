/* Regression for a defect found in review of the invariant synthesiser; see
 * test.desc for the flags. Without the fix this program is mis-verified. */
#include <stdint.h>
#include <assert.h>
int main(void) {
  uint32_t n; uint64_t a; int c;
  __ESBMC_assume(n >= 1 && n <= 3);
  uint64_t s = 0;
  if (c) s = 5;
  uint64_t i = 1;
  while (i <= n) { s = s + a; i++; }
  assert(s == (c ? 5 : 0) + (uint64_t)n * a);
  return 0;
}
