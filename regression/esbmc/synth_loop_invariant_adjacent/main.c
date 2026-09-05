/* Regression for a defect found in review of the invariant synthesiser; see
 * test.desc for the flags. Without the fix this program is mis-verified. */
#include <stdint.h>
#include <assert.h>
int main(void) {
  uint32_t n; uint64_t a; __ESBMC_assume(n >= 1 && n <= 3);
  uint64_t i = 1, sn = 0;
  while (i <= n) { sn = sn + a; i++; }
  uint64_t j = 0, t = 1;
  while (j < 4) { t = t * 3; j++; }
  assert(t == 81);
  return 0;
}
