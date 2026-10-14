/* The bit-vector encoding builds a concrete NaN bit-pattern, so it keeps the
 * sign the native floating-point theory cannot represent (#7021). */
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
