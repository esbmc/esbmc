#include <assert.h>
#include <math.h>

int main(void)
{
  volatile double m0 = -0.0, p0 = 0.0, tiny = 1e-20, one = 1.0, h = 0.5;
  assert(signbit(atanh(m0)) && atanh(m0) == 0.0);
  assert(!signbit(atanh(p0)));
  assert(atanh(tiny) == 1e-20 && atanh(-tiny) == -1e-20);
  assert(atanh(one) == INFINITY && atanh(-one) == -INFINITY);
  assert(isnan(atanh(2.0)));
  assert(atanh(-h) == -atanh(h));
  assert(atanh(h) > 0.5493 && atanh(h) < 0.5494);
  return 0;
}
