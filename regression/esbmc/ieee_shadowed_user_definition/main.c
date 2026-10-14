#include <assert.h>

/* #6904: a program is free to define fma, remainder or nearbyint itself, so its
 * body is what must be verified, not the IEEE node. */
double fma(double a, double b, double c)
{
  return 42.0;
}

double remainder(double a, double b)
{
  return 43.0;
}

double nearbyint(double a)
{
  return 44.0;
}

int main(void)
{
  assert(fma(2.0, 3.0, 4.0) == 42.0);
  assert(remainder(5.0, 3.0) == 43.0);
  assert(nearbyint(2.5) == 44.0);
  return 0;
}
