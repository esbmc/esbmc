/* RTE tie at 1.5: floor=1 (odd) so round up to 2 */
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x == 1.5);
  double y = nearbyint(x);
  __ESBMC_assert(y == 2.0, "RTE tie at 1.5 rounds to 2");
  return 0;
}
