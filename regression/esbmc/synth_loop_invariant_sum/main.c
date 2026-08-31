/* Affine accumulator: k-induction with --interval-analysis cannot prove this.
 * The interval domain is non-relational, so at the loop head it knows only
 * i in [1, UINT64_MAX] and nothing tying sn to i and a.
 * --synthesise-loop-invariants derives sn == (i - 1) * a. */
#include <stdint.h>
#include <assert.h>

int main(void)
{
  uint64_t i = 1;
  uint64_t sn = 0;
  uint32_t n;
  uint64_t a;

  __ESBMC_assume(n >= 1);

  while (i <= n)
  {
    sn = sn + a;
    i++;
  }

  assert(sn == (uint64_t)n * a);
  return 0;
}
