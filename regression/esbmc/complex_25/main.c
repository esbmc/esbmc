#include <assert.h>

int calls = 0;

_Complex double f(void)
{
  calls++;
  return 1.0 + 2.0i;
}

int dcalls = 0;
double d(void)
{
  dcalls++;
  return 2.0;
}

int depth = 0;
_Complex double g(void)
{
  if (depth++ > 1)
    return 1.0 + 1.0i;
  _Complex double inner = -g();
  return inner;
}

int main()
{
  _Complex double z = 3.0 + 4.0i;

  _Complex double s = f() + z;
  assert(calls == 1);
  assert(__real__ s == 4.0 && __imag__ s == 6.0);

  _Complex double n = -f();
  assert(calls == 2);
  assert(__real__ n == -1.0 && __imag__ n == -2.0);

  _Complex double c = ~f();
  assert(calls == 3);
  assert(__real__ c == 1.0 && __imag__ c == -2.0);

  _Complex double m = f() * f();
  assert(calls == 5);
  assert(__real__ m == -3.0 && __imag__ m == 4.0);

  _Complex double p = z * d();
  assert(dcalls == 1);
  assert(__real__ p == 6.0 && __imag__ p == 8.0);

  _Complex double r = -g();
  assert(depth == 3);
  assert(__real__ r == -1.0 && __imag__ r == -1.0);

  return 0;
}
