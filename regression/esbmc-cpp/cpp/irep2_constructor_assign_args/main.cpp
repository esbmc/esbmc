#include <cassert>

extern "C" int nondet_int();

// The member initialiser reaches the adjuster as `m = T(x, y)`, whose call is
// missing the object argument: converting its arguments before the fold
// inserts `&m` matches each against the wrong parameter, and the double is
// round-tripped through int.
struct T
{
  int a;
  double b;
  T(int x, double y) : a(x), b(y)
  {
  }
};

struct L
{
  T m;
  L(int x, double y) : m(x, y)
  {
  }
};

int main()
{
  int i = nondet_int();
  L l(i, 2.5);
  assert(l.m.a == i && l.m.b == 2.5);
}
