/* |fmod(x,y)| < |y| -- true for a correct fmod; the solver must prove it
 * across the domain, exercising fp_convt's fp.rem encoding. */
#include <assert.h>

double __VERIFIER_nondet_double(void);
#include <math.h>
int main(void)
{
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  __ESBMC_assume(isgreaterequal(x, -1e6) && islessequal(x, 1e6));
  __ESBMC_assume(isgreaterequal(y, 1.0) && islessequal(y, 1024.0));
  double r = fmod(x, y);
  assert(isless(fabs(r), fabs(y)));
  return 0;
}
