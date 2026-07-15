/* RTA tie at -0.5: negative, round away from zero -> -1 */
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x == -0.5);
  double y = nearbyint(x);
  __ESBMC_assert(y == -1.0, "RTA: tie at -0.5 rounds away from zero to -1");
  return 0;
}
