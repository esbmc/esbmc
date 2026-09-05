/* Affine accumulator with a SYMBOLIC addend: k-induction with
 * --interval-analysis cannot prove this. The interval domain is non-relational,
 * so at the loop head it knows only i's range and nothing tying sn to i and a.
 * --synthesise-loop-invariants derives sn == (i - 1) * a.
 *
 * Widths are deliberately small. The symbolic-addend closed form makes the exit
 * obligation a multiplier-equivalence miter, and at 64 bits only bitwuzla
 * discharges it -- z3 does not finish in 240s, which is what the Z3-only
 * Windows leg runs. The 64-bit form is pinned by
 * regression/bitwuzla/synth_loop_invariant_sum64. */
#include <stdint.h>
#include <assert.h>

int main(void)
{
  uint32_t i = 1;
  uint32_t sn = 0;
  uint8_t n;
  uint8_t a;

  __ESBMC_assume(n >= 1 && n <= 15);

  while (i <= n)
  {
    sn = sn + a;
    i++;
  }

  assert(sn == (uint32_t)n * a);
  return 0;
}
