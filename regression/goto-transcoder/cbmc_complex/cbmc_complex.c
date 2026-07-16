#include <assert.h>

int main() {
  _Complex double a = 1.0 + 2.0i;
  _Complex double b = 3.0 - 1.0i;

  assert(__real__ a == 1.0 && __imag__ a == 2.0);

  _Complex double s = a + b;
  assert(__real__ s == 4.0 && __imag__ s == 1.0);

  _Complex double d = a - b;
  assert(__real__ d == -2.0 && __imag__ d == 3.0);

  _Complex double m = a * b;
  assert(__real__ m == 5.0 && __imag__ m == 5.0);

  _Complex double q = m / b;
  assert(q == a);

  _Complex double p = a * 2.0;
  assert(__real__ p == 2.0 && __imag__ p == 4.0);

  _Complex double n = -a;
  assert(__real__ n == -1.0 && __imag__ n == -2.0);


  _Complex double t3 = a + b + n;
  assert(__real__ t3 == 3.0 && __imag__ t3 == -1.0);

  double rp = (double)a;
  assert(rp == 1.0);

  _Complex float f = (_Complex float)a;
  assert(__real__ f == 1.0f && __imag__ f == 2.0f);

  _Complex float w = f + f;
  assert(__real__ w == 2.0f && __imag__ w == 4.0f);

  _Complex int ci = 2 + 3i;
  _Complex int cm = ci * (1 - 1i);
  assert(__real__ cm == 5 && __imag__ cm == 1);

  return 0;
}
