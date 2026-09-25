/* SMT-LIB's FP theory has a single, sign-less NaN, so under the native
 * floating-point theory the sign of a NaN is unconstrained and this fails.
 * ESBMC's own bit-vector encoding, --fp2bv, gets it right. See #7021. */
#include <assert.h>
#include <math.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double m = __VERIFIER_nondet_double();
  __ESBMC_assume(m < 0.0);
  double s = copysign(NAN, m);
  assert(isnan(s) && signbit(s));
  return 0;
}
