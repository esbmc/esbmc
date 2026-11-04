#include <assert.h>
#include <math.h>

int main(void)
{
  double _Complex z;
  __real__ z = 1.0;
  __imag__ z = INFINITY;
  double x = 2.0;
  double _Complex r = x * z;
  assert(isnan(__real__ r));
  return 0;
}
