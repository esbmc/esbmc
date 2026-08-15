/* |remainder(x,y)| <= |y|/2 -- IEEE-754 remainder bound, C17 7.12.10.2.
 *
 * y is constant and x symbolic over a wide range. With both operands symbolic
 * this bound does not solve inside the suite's 8 GiB cap: fp.rem aligns the
 * significands across the whole exponent range, so the encoding carries a
 * ~2100-bit bvudiv. 3.0 is not a power of two, so the division is still
 * exercised rather than collapsing to a shift; the symbolic-symbolic pair is
 * covered by ieee_rem_fmod_bound, whose |y| bound does fit. */
#include <assert.h>

double __VERIFIER_nondet_double(void);
#include <math.h>
int main(void)
{
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(isgreaterequal(x, -1e6) && islessequal(x, 1e6));
  double r = remainder(x, 3.0);
  assert(islessequal(fabs(r), 1.5));
  return 0;
}
