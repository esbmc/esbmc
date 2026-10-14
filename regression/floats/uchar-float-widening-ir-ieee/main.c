/* Regression: unsigned char range must be respected under --ir-ieee.
 *
 * An unsigned char x is compared to a nondet float f via f == x, which
 * implicitly promotes x to float.  Since all unsigned char values [0, 255]
 * are exactly representable in float32, the float that equals x is exactly x.
 * Widening that float to double (d = f) is lossless, so d == x must hold
 * whenever f == x.
 *
 * Under --ir-ieee, unsigned char variables were formerly encoded as unbounded
 * Z3 integers, allowing the solver to pick values outside [0, 255] and
 * exploit the int-to-float rounding asymmetry to generate a spurious
 * counterexample.  The fix asserts 0 <= x <= 255 in the SMT formula so that
 * only valid C values are considered. */

#include <assert.h>

extern float __VERIFIER_nondet_float(void);
extern unsigned char __VERIFIER_nondet_uchar(void);

int main(void)
{
  float f = __VERIFIER_nondet_float();
  unsigned char x = __VERIFIER_nondet_uchar();
  double d = f;

  if (f == x)
    assert(d == x);

  return 0;
}
