/* #9350: a float sum whose first operand is a constant was printed in the
 * counterexample as that constant instead of the sum. */
#include <assert.h>
float nondet_float(void);
double nondet_double(void);

int main(void)
{
  float h = nondet_float();
  __ESBMC_assume(h >= 1.0f && h <= 1.0f);
  double g = nondet_double();
  __ESBMC_assume(g >= 2.0 && g <= 2.0);
  float a = 0.25f + h;
  float s = 0.25f - h;
  float m = 0.25f * h;
  float d = 0.5f / h;
  double e = 0.25 + g;
  assert(a + s + m + d + e < 0.0);
  return 0;
}
