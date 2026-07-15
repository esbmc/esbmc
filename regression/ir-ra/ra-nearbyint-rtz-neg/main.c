/* RTZ (truncate toward 0): x in (-2,-1) -> -1 */
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 4; /* ROUND_TO_ZERO */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x > -2.0 && x < -1.0);
  double y = nearbyint(x);
  __ESBMC_assert(y == -1.0, "RTZ: nearbyint of (-2,-1) is -1");
  return 0;
}
