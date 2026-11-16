/* #9362: C17 6.5.9p4 compares complex values part by part as reals. */
#include <assert.h>
#include <math.h>

int g;
double _Complex f(void)
{
  g++;
  return -0.0;
}

int main(void)
{
  double _Complex a = 0.0, b = -0.0;
  assert(a == b);
  assert(!(a != b));

  double _Complex z = 2.0;
  float _Complex w = 2.0f;
  int _Complex iz = 3;
  assert(z == 2.0);
  assert(2 == z);
  assert(z == w);
  assert(iz == 3);
  assert(f() == 0.0);
  assert(g == 1);

  double _Complex n = NAN;
  assert(n != n);
  assert(!(n == 1.0));
  return 0;
}
