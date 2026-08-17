/* remquo's remainder obeys the same |r| <= |y|/2 bound as remainder().
 * Single precision for the same reason as ieee_rem_remainder_bound. */
#include <assert.h>

float __VERIFIER_nondet_float(void);
#include <math.h>
int main(void)
{
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  __ESBMC_assume(isgreaterequal(x, -1e6f) && islessequal(x, 1e6f));
  __ESBMC_assume(isgreaterequal(y, 1.0f) && islessequal(y, 1024.0f));
  int q;
  float r = remquof(x, y, &q);
  assert(islessequal(fabsf(r), fabsf(y) * 0.5f));
  return 0;
}
