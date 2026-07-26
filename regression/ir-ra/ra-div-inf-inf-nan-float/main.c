#include <math.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  float x = __VERIFIER_nondet_float();
  __ESBMC_assume(isinf(x));
  float y = __VERIFIER_nondet_float();
  __ESBMC_assume(isinf(y));
  float z = x / y;

  __ESBMC_assert(isnan(z), "isinf(x) && isinf(y) implies isnan(x / y)");
  return 0;
}
