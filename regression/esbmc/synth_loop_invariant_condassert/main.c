/* An assert behind a branch leaves an ASSERT inside a goto diamond in the loop
 * body, rather than the single ASSERT a top-level assert folds to. The body
 * scan steps over such a region as a unit -- it writes nothing, so it cannot
 * make the per-iteration effect conditional -- and the loop still summarises.
 * Refusing the diamond declines every loop that asserts under a condition. */
#include <assert.h>
#include <stdint.h>

int main(void)
{
  uint32_t n;
  uint64_t a;
  uint64_t i = 1, sn = 0;
  __ESBMC_assume(n >= 1 && n <= 10);
  __ESBMC_assume(a <= 10);
  while (i <= n)
  {
    if (i > 1)
      assert(sn >= a);
    sn = sn + a;
    i++;
  }
  return 0;
}
