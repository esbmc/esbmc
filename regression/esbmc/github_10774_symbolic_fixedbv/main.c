// For every x and nonzero y: fmod has x's sign and |fmod| < |y|, remainder
// lies in [-|y|/2, |y|/2], and the two differ by 0 or |y|.
#include <math.h>
int main()
{
  double x = __VERIFIER_nondet_double(), y = __VERIFIER_nondet_double();
  __ESBMC_assume(y != 0.0 && y > -1000.0 && y < 1000.0);
  double ay = y < 0.0 ? -y : y;
  double f = fmod(x, y);
  double r = remainder(x, y);
  double af = f < 0.0 ? -f : f, ar = r < 0.0 ? -r : r;
  __ESBMC_assert(af < ay, "fmod bound");
  __ESBMC_assert(f == 0.0 || (f < 0.0) == (x < 0.0), "fmod sign");
  __ESBMC_assert(ar <= ay - ar, "remainder bound");
  __ESBMC_assert(r == f || r - f == ay || f - r == ay, "remainder vs fmod");
  return 0;
}
