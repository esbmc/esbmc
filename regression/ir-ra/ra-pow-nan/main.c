#include <math.h>

int main(void)
{
  /* The operational model defines pow(negative, non-integer) as NaN.
   * --unwind 1 bounds the exp/expm1 path that is unreachable for this
   * concrete input; unwinding assertions remain active by default. */
  double r = pow(-2.0, 0.5);
  __ESBMC_assert(r != r, "pow(-2.0, 0.5) must be NaN");
  return 0;
}
