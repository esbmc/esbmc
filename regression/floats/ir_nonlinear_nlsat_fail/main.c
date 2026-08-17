/* A nested product of unbounded floats under directed rounding: nonlinear in
 * the reals the integer/real encoding uses, so Z3's `smt` tactic alone reports
 * "incomplete (theory arithmetic)" and no verdict is reached at all.
 * The violation is a NaN operand, for which the equality is false. */
#include <assert.h>

extern int __ESBMC_rounding_mode;

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF */
  float x = nondet_float();
  float y = nondet_float();
  float z = x * y;
  float w = z * z;

  assert(w == z * z);
  return 0;
}
