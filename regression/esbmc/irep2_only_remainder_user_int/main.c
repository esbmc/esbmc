#include <assert.h>

/* Not floating-point: ieee_rem2t is a floatbv node, so the lowering must
 * decline on an integer-shaped declaration of the same name. */
int remainder(int, int);
double nearbyint(double);

int main(void)
{
  int r = remainder(7, 3);
  assert(r == r);
  assert(nearbyint(2.5) == 2.0);
  return 0;
}
