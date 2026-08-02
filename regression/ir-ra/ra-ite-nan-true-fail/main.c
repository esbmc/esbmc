#include <math.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  __ESBMC_rounding_mode = 0;
  double x = __VERIFIER_nondet_double();
  __ESBMC_assume(x == -1.0);
  double y = sqrt(x);
  int c = __VERIFIER_nondet_int();
  __ESBMC_assume(c != 0);
  double z = c ? y : 0.0;
  __ESBMC_assert(!isnan(z), "true branch NaN: z must not be NaN");
  return 0;
}
