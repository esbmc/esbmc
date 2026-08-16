#include <math.h>
#include <assert.h>

double nondet_double(void);

/* fmod(1e18, 3.0) is exactly 1.0: 1e18 is an integer in double, and
 * 10 == 1 (mod 3), so 10^18 == 1 (mod 3).
 *
 * The model computes x - y * (int)(x / y) (src/c2goto/library/libm/fmod.c).
 * Here x / y is about 3.3e17, so the (int) conversion is undefined -- the
 * quotient does not fit -- and the result is unrelated to the remainder.
 * The values are read through nondet_double so the frontend cannot fold the
 * call; written as literals, clang evaluates fmod at compile time and the
 * model is never exercised.
 *
 * A correct fix needs an exact remainder at the SMT layer (fp.rem), which
 * ESBMC's fp_convt does not expose yet -- see esbmc/esbmc#6896. */
int main(void)
{
  double x = nondet_double();
  double y = nondet_double();
  __ESBMC_assume(x == 1e18);
  __ESBMC_assume(y == 3.0);

  assert(fmod(x, y) == 1.0);
  return 0;
}
