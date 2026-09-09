/* |remainder(x,y)| <= |y|/2 -- IEEE-754 remainder bound, C17 7.12.10.2. */
#include <assert.h>

double __VERIFIER_nondet_double(void);
#include <math.h>
int main(void)
{
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  __ESBMC_assume(isgreaterequal(x, -1e6) && islessequal(x, 1e6));
  __ESBMC_assume(isgreaterequal(y, 1.0) && islessequal(y, 1024.0));
  double r = remainder(x, y);
  assert(islessequal(fabs(r), fabs(y) * 0.5));
  return 0;
}
