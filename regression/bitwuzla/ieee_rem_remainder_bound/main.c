/* |remainder(x,y)| <= |y|/2 -- IEEE-754 remainder bound, C17 7.12.10.2.
 * Single precision: the bound is format-independent, and the lowering's
 * significand alignment spans 2^ebits - 3 bits, so double makes the division
 * ~2100 bits wide and exhausts the regression memory cap. ieee_rem_fmod_sign
 * still exercises the double path. */
#include <assert.h>

float __VERIFIER_nondet_float(void);
#include <math.h>
int main(void)
{
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  __ESBMC_assume(isgreaterequal(x, -1e6f) && islessequal(x, 1e6f));
  __ESBMC_assume(isgreaterequal(y, 1.0f) && islessequal(y, 1024.0f));
  float r = remainderf(x, y);
  assert(islessequal(fabsf(r), fabsf(y) * 0.5f));
  return 0;
}
