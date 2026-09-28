/* RTN (round toward -inf): x in (-2,-1) -> -2 */
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 3; /* ROUND_TO_MINUS_INF */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > -2.0 && x < -1.0);
  double y = nearbyint(x);
  __ESBMC_assert(y == -2.0, "RTN: nearbyint of (-2,-1) is -2");
  return 0;
}
