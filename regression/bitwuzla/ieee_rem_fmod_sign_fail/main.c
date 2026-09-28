/* Negation of the C17 7.12.10.1 sign rule: a correct fmod refutes it. */
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
  assert(r == 0.0 || (signbit(r) != signbit(x)));
  return 0;
}
