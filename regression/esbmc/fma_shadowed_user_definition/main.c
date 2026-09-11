#include <assert.h>

/* #6904: a program is free to define fma, remainder or nearbyint itself, so
 * its body is what must be verified, not the IEEE node. */
double fma(double a, double b, double c)
{
  return 42.0;
}

int main(void)
{
  assert(fma(2.0, 3.0, 4.0) == 42.0);
  return 0;
}
