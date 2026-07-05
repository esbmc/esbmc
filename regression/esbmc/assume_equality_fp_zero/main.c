#include <assert.h>
extern void __ESBMC_assume(_Bool);
extern double nondet_double(void);

int main()
{
  double x = nondet_double();
  __ESBMC_assume(x == 0.0);
  // Signbit-sensitive: 1.0/+0.0 = +inf, 1.0/-0.0 = -inf.
  // If the lift wrote level2[x] = +0.0, the assertion would fold to
  // "+inf > 0" -> trivially true, missing the bug. With the guard the
  // solver picks x = -0.0 and the assertion fails.
  assert(1.0 / x > 0);
  return 0;
}
