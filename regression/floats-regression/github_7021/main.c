// copysign(NAN, -2.0) is a NaN carrying the sign of -2.0, and signbit is
// nonzero iff the sign is negative -- for a NaN as much as for a finite value
// (C17 7.12.11.1, 7.12.3.6). The native-FP backend left the sign of a NaN
// unconstrained, so this held under --fp2bv but not under --z3 (#7021).
#include <assert.h>
#include <math.h>

int main(void)
{
  double s = copysign(NAN, -2.0);
  assert(isnan(s));
  assert(signbit(s));
  return 0;
}
