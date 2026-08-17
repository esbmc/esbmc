#include <math.h>
#include <assert.h>

double nondet_double(void);

/* fmod(1e18, 3.0) is exactly 1.0: 1e18 is an integer in double, and
 * 10 == 1 (mod 3), so 10^18 == 1 (mod 3).
 *
 * The old model computed x - y * (int)(x / y); here x / y is about 3.3e17,
 * so the (int) conversion was undefined and the result unrelated to the
 * remainder. fmod now rides the solver's exact fp.rem (esbmc/esbmc#6896).
 * The values are read through nondet_double so the frontend cannot fold the
 * call; written as literals, clang evaluates fmod at compile time and the
 * model is never exercised. */
int main(void)
{
  double x = nondet_double();
  double y = nondet_double();
  __ESBMC_assume(x == 1e18);
  __ESBMC_assume(y == 3.0);

  assert(fmod(x, y) == 1.0);
  return 0;
}
