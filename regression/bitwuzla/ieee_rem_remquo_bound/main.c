/* remquo's remainder obeys the same |r| <= |y|/2 bound as remainder().
 * Constant divisor for the reason given in ieee_rem_remainder_bound. */
#include <assert.h>

double __VERIFIER_nondet_double(void);
#include <math.h>
int main(void)
{
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(isgreaterequal(x, -1e6) && islessequal(x, 1e6));
  int q;
  double r = remquo(x, 3.0, &q);
  assert(islessequal(fabs(r), 1.5));
  return 0;
}
