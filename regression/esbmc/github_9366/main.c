#include <assert.h>
#include <math.h>

int main(void)
{
  double _Complex z, m;
  __real__ z = 1.0;
  __imag__ z = INFINITY;
  __real__ m = 1.0;
  __imag__ m = -0.0;
  double x = 2.0, y = 1.0;

  double _Complex r = x * z;
  assert(__real__ r == 2.0 && isinf(__imag__ r));
  r = z / x;
  assert(__real__ r == 0.5 && isinf(__imag__ r));
  r = y + m;
  assert(__real__ r == 2.0 && signbit(__imag__ r));
  r = y - m;
  assert(__real__ r == 0.0 && !signbit(__imag__ r));
  r = m + y;
  assert(__real__ r == 2.0 && signbit(__imag__ r));
  r = x / m;
  assert(__real__ r == 2.0 && !signbit(__imag__ r));
  r = m;
  r -= y;
  assert(__real__ r == 0.0 && signbit(__imag__ r));
  return 0;
}
