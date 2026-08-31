/* The synthesised invariant is established and preserved, but the property is
 * genuinely false (off by one). Synthesis must not mask the violation. */
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

  assert(sn == (uint64_t)n * a + 1);
  return 0;
}
