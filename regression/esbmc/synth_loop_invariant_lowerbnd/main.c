/* Regression for a defect found in review of the invariant synthesiser; see
 * test.desc for the flags. Without the fix this program is mis-verified. */
#include <stdint.h>
#include <assert.h>
int main(void) {
  uint32_t n; uint64_t a; uint64_t i = 1, sn = 0;
  __ESBMC_assume(n >= 1 && n <= 10); __ESBMC_assume(a <= 10);
  while (i <= n) { assert(sn <= (uint64_t)n * a); sn = sn + a; i++; }
  return 0;
}
