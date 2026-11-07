#include <assert.h>

/* #10007: a declaration whose arity does not match the C library function is
 * an ordinary call, not the fixed-arity float node. */
double fma(double, double);
double nearbyint(void);
double remainder(double);

int main(void)
{
  double f = fma(2.0, 3.0);
  double n = nearbyint();
  double r = remainder(7.0);
  assert(f == f || f != f);
  assert(n == n || n != n);
  assert(r == r || r != r);
  return 0;
}
