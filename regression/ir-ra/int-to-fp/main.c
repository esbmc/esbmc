#include <assert.h>

extern int __VERIFIER_nondet_int(void);

/* Under --ir-ra, a nondet int lifted to double can be <= 0.0,
 * so the assertion must fail. */
int main(void)
{
  int n = __VERIFIER_nondet_int();
  double x = (double)n;
  assert(x > 0.0);
  return 0;
}
