/* Failing companion: RTE rounds x in (0.5,1.5) to 1, so assert != 1 must fail */
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > 0.5 && x < 1.5);
  double y = nearbyint(x);
  __ESBMC_assert(y != 1.0, "should be unreachable");
  return 0;
}
