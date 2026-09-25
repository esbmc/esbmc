#include <float.h>
#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  /* fmod(finite, finite nonzero): well-defined, not NaN */
  __ESBMC_assume(x > 0.0 && x < DBL_MAX);
  __ESBMC_assume(y > 0.0 && y < DBL_MAX);
  double r = fmod(x, y);

  __ESBMC_assert(r == r, "fmod(finite, finite) should not be NaN");
  return 0;
}
