/* Companion to ir_nonlinear_nlsat_fail: the same nonlinear goal with the NaN
 * operands assumed away, so the equality holds and the nonlinear fallback must
 * not manufacture a counterexample. */
#include <assert.h>

extern int __ESBMC_rounding_mode;

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF */
  float x = nondet_float();
  float y = nondet_float();
  float z = x * y;
  float w = z * z;

  __ESBMC_assume(z == z && w == w);

  assert(w == z * z);
  return 0;
}
