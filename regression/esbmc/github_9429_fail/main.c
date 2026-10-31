#include <assert.h>
#include <math.h>

int main(void)
{
  volatile double x = -0.0;
  double r = atanh(x);
  assert(!signbit(r));
  return 0;
}
