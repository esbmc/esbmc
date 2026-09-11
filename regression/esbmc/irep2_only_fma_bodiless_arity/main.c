#include <assert.h>

/* Wrong arity for the C library fma: the lowering must decline rather than
 * build a three-operand node out of two arguments. */
double fma(double, double);
double nearbyint(double);

int main(void)
{
  double r = fma(2.0, 3.0);
  assert(r == r || r != r);
  assert(nearbyint(2.5) == 2.0);
  return 0;
}
