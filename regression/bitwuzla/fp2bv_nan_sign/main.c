/* The bit-vector encoding builds a concrete NaN bit-pattern, so it keeps the
 * sign the native floating-point theory cannot represent (#7021). */
#include <assert.h>
#include <math.h>
int main(void)
{
  double s = copysign(NAN, -2.0);
  assert(isnan(s) && signbit(s));
  return 0;
}
