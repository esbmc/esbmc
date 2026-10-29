#include <assert.h>
#include <math.h>

int main(void)
{
  volatile double x = -0.0;
  double r = atan(x);
  assert(!signbit(r));
  return 0;
}
