/* #9362 under --clang-c-irep2-adjust-only. */
#include <assert.h>
#include <math.h>

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

  double _Complex n = NAN;
  assert(n != n);
  assert(!(n == 1.0));
  return 0;
}
