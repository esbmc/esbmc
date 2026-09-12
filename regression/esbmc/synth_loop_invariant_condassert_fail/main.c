/* The counterpart of synth_loop_invariant_condassert: the same diamond, with a
 * claim the closed form does not imply. Recognising the region must not make
 * the abstraction prove it -- the cut loop reports the claim unknown rather
 * than passed, so a body whose assert is a branch cannot be a false proof. */
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
      assert(sn > a);
    sn = sn + a;
    i++;
  }
  return 0;
}
