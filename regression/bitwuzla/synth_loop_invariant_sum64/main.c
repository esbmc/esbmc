/* The 64-bit symbolic-addend shape the two-disjunct bound was designed for.
 * Negating the guard at the exit yields the equality i == E by disjunct
 * elimination, letting the two `* a` terms share a multiplier; the inequality
 * form leaves the solver proving two 64-bit multipliers equivalent and does not
 * terminate. Only bitwuzla discharges the miter at this width, hence the pin
 * and the suite -- regression/esbmc/synth_loop_invariant_sum carries the same
 * shape narrowed so every platform runs it. */
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
