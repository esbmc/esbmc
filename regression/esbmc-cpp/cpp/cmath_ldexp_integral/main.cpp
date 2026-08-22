#include <cmath>
#include <cassert>

int main()
{
  // [cmath.syn]: an integer argument converts to double. Without that
  // overload the call is ambiguous between float, double and long double.
  int mant = 3;
  assert(std::ldexp(mant, 2) == 12.0);

  unsigned u = 5;
  assert(std::ldexp(u, 1) == 10.0);

  long l = 7;
  assert(std::ldexp(l, 0) == 7.0);

  // The floating-point overloads must still win for floating arguments.
  assert(std::ldexp(1.5, 2) == 6.0);
  return 0;
}
