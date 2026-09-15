#include <assert.h>

/* ieee_fma2t is homogeneous in its own type, so a declaration mixing widths
 * must decline rather than reach the solver with a mismatched operand. */
long double fmal(double, double, double);
double nearbyint(double);

int main(void)
{
  long double r = fmal(2.0, 3.0, 4.0);
  if (r == 10.0L)
    assert(r == 10.0L);
  assert(nearbyint(2.5) == 2.0);
  return 0;
}
