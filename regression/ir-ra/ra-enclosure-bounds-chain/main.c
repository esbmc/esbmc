/* Regression: --ir-ra should process a small chain of double-precision
 * operations through the enclosure path. The assertion is intentionally false
 * so ESBMC produces a counterexample and exercises multiple enclosure sites. */
#include <assert.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double a = __VERIFIER_nondet_double();
  double b = __VERIFIER_nondet_double();
  double c = __VERIFIER_nondet_double();

  double t = a * b; /* first enclosure pair generated here */
  double s = t + c; /* second enclosure pair generated here */

  /* Falsifiable: a=1, b=1, c=0 gives s=1, which is not > 100. */
  assert(s > 100.0);
  return 0;
}
