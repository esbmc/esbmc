/* SMT-LIB's FP theory has a single, sign-less NaN, so under
 * --bitwuzla-native-fp the sign of a NaN is unconstrained and this fails.
 * ESBMC's own bit-vector encoding, the default, gets it right. See #7021. */
#include <assert.h>
#include <math.h>
int main(void)
{
  double s = copysign(NAN, -2.0);
  assert(isnan(s) && signbit(s));
  return 0;
}
