/* Same program as synth_loop_invariant_sum, run with --interval-analysis.
 * That pass rewrites the loop head into the simplified `IF i > n GOTO exit`,
 * which has no not2t to strip, and moves the back-edge target ahead of the
 * guard; the recogniser must still match, and the marker must still land where
 * goto_loop_invariant's extractor searches. */
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
