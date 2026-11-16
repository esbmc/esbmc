/* #9371: an infinity times nonzero finite numbers stays an infinity
   (C17 G.5.1p4) along a chained product, whose lowering must not grow
   exponentially with the chain. */
#include <assert.h>
#include <math.h>

int main(void)
{
  double _Complex a, b, c, d;
  __real__ a = INFINITY;
  __imag__ a = 0.0;
  __real__ b = 1.0;
  __imag__ b = 0.0;
  __real__ c = 0.5;
  __imag__ c = -2.0;
  __real__ d = -3.0;
  __imag__ d = 1.0;
  double _Complex r = (((((a * b) * c) * d) * b) * c) * d;
  assert(isinf(__real__ r) || isinf(__imag__ r));
  r = (((((a / b) / c) / d) / b) / c) / d;
  assert(isinf(__real__ r) || isinf(__imag__ r));
  return 0;
}
