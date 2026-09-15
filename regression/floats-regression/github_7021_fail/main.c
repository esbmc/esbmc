// Negative counterpart of github_7021: the NaN's sign comes from -2.0, so
// signbit is nonzero and this assertion must be refuted rather than held
// vacuously by an unconstrained sign.
#include <assert.h>
#include <math.h>

int main(void)
{
  double s = copysign(NAN, -2.0);
  assert(isnan(s));
  assert(!signbit(s));
  return 0;
}
