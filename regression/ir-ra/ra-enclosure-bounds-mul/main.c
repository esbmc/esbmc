/* Regression: --ir-ra should process a double-precision multiplication through
 * the enclosure path. The assertion is intentionally false so ESBMC produces a
 * counterexample and exercises the encoding. */
#include <assert.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double a = __VERIFIER_nondet_double();
  double b = __VERIFIER_nondet_double();
  double p = a * b;

  /* Falsifiable: a=0, b=anything gives p=0 which is not > a+b in general. */
  assert(p > a + b + 1.0);
  return 0;
}
