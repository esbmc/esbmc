/* Already-integral input: nearbyint(3.0) == 3.0 under any rounding mode */
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x == 3.0);
  double y = nearbyint(x);
  __ESBMC_assert(y == 3.0, "nearbyint of an integer returns that integer");
  return 0;
}
