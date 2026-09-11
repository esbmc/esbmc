#include <assert.h>

/* Each node is fixed-arity, so a declaration that does not match must decline
 * rather than index past the end of the argument list. */
double fma(double, double);
double nearbyint(void);
double remainder(double);
double sqrt(double);

int main(void)
{
  double f = fma(2.0, 3.0);
  double n = nearbyint();
  double r = remainder(7.0);
  assert(f == f || f != f);
  assert(n == n || n != n);
  assert(r == r || r != r);
  assert(sqrt(4.0) == 2.0);
  return 0;
}
