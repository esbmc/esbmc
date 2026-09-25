/* |remainder(x,y)| <= |y|/2 -- IEEE-754 remainder bound, C17 7.12.10.2.
 * Single precision, constant divisor: the bound is format-independent, but the
 * lowering aligns significands across 2^ebits - 3 bits, so a symbolic divisor
 * makes the division exhaust the regression memory cap at either width. 3.0f
 * is not a power of two, so the division is still exercised rather than
 * collapsing to a shift. ieee_rem_fmod_sign still drives the double path, and
 * ieee_rem_remquo_bound keeps a symbolic divisor. */
#include <assert.h>

float __VERIFIER_nondet_float(void);
#include <math.h>
int main(void)
{
  float x = __VERIFIER_nondet_float();
  __ESBMC_assume(isgreaterequal(x, -1e6f) && islessequal(x, 1e6f));
  float r = remainderf(x, 3.0f);
  assert(islessequal(fabsf(r), 1.5f));
  return 0;
}
