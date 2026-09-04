/* Regression for a defect found in review of the invariant synthesiser; see
 * test.desc for the flags. Without the fix this program is mis-verified.
 *
 * --unwind 12 is inert when synthesis fires: the loop is havoc'd, so there is
 * nothing to unwind and the verdict is unchanged. It is there for when
 * synthesis *declines* -- n is symbolic, so the loop then unwinds without
 * bound and the test burns the full 1200s ctest cap rather than reporting
 * anything. 11 iterations suffice for n <= 10. The test still fails in that
 * case, on the missing "Synthesised loop invariants" line, which names the
 * defect instead of timing out. */
#include <stdint.h>
#include <assert.h>
int main(void) {
  uint32_t n; uint64_t a; uint64_t i = 1, sn = 0;
  __ESBMC_assume(n >= 1 && n <= 10); __ESBMC_assume(a <= 10);
  while (i <= n) { assert(sn <= (uint64_t)n * a); sn = sn + a; i++; }
  return 0;
}
