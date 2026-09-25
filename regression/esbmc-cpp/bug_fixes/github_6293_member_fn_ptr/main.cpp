// github #6293: calling a pointer-to-member function that takes arguments must
// pass the implicit object as `this` (the first parameter) and the explicit
// arguments after it. Previously `this` was appended at the back of the
// member-pointer's parameter list, shifting every explicit argument by one.
#include <cassert>

struct S
{
  int v;
  int g(int x) { return v + x; }
  int h(int x, int y) { return v + x * 10 + y; }
};

int main()
{
  S s;
  s.v = 100;

  int (S::*p)(int) = &S::g;
  assert((s.*p)(5) == 105);

  S *q = &s;
  assert((q->*p)(7) == 107);

  int (S::*r)(int, int) = &S::h;
  assert((s.*r)(3, 4) == 134);
  return 0;
}
